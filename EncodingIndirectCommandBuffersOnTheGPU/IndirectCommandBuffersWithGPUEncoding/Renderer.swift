import Foundation
import MetalKit

private let maxFramesInFlight = 3

private enum MovementDirection: Int {
    case right
    case up
    case left
    case down
}

private let AAPLGridHeight = ((AAPLNumObjects + AAPLGridWidth - 1) / AAPLGridWidth)

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderPipelineState: MTLRenderPipelineState
    private let computePipelineState: MTLComputePipelineState

    private let inFlightSemaphore = DispatchSemaphore(value: maxFramesInFlight)

    private let vertexBuffer: MTLBuffer
    private let objectParameters: MTLBuffer
    private let frameStateBuffer: [MTLBuffer]
    private let indirectCommandBuffer: MTLIndirectCommandBuffer
    private let icbArgumentBuffer: MTLBuffer

    private var inFlightIndex = 0
    private var frameNumber = 0

    private var gridCenter: SIMD2<Float> = [0, 0]
    private var movementSpeed: Float = 0.15
    private var objectDirection: MovementDirection = .up

    private var aspectScale: SIMD2<Float> = [0, 0]

    static private let gridDimensions: SIMD2<Float> = [Float(AAPLGridWidth),
                                                       Float(AAPLGridHeight)]

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.25, green: 0.25, blue: 0.25, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        metalView.depthStencilPixelFormat = .depth32Float

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "MyPipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat
        // Needed for this pipeline state to be used in indirect command buffers
        pipelineStateDescriptor.supportIndirectCommandBuffers = true

        self.renderPipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        guard let gpuCommandEncodingKernel = defaultLibrary.makeFunction(name: "cullMeshesAndEncodeCommands") else {
            fatalError()
        }

        self.computePipelineState = {
            do {
                return try device.makeComputePipelineState(function: gpuCommandEncodingKernel)
            } catch {
                fatalError("Failed to create compute pipeline state: \(error)")
            }
        }()

        let tempMeshes: [[AAPLVertex]] = (0..<AAPLNumObjects).map { _ in
            let numTeeth = Int.random(in: 3...53)
            let innerRatio = Float.random(in: 0.2...0.9)
            let toothWidth = Float.random(in: 0.1...0.5)
            let toothSlope = Float.random(in: 0.0...0.2)

            return Renderer.makeGearMesh(numTeeth: numTeeth,
                                         innerRatio: innerRatio,
                                         toothWidth: toothWidth,
                                         toothSlope: toothSlope)
        }

        // Create and fill array containing parameters for each object
        let objectParameterArraySize = Int(AAPLNumObjects) * MemoryLayout<AAPLObjectParameters>.stride

        guard let paramBuffer = device.makeBuffer(length: objectParameterArraySize) else { fatalError() }
        objectParameters = paramBuffer
        objectParameters.label = "Object Parameters Array"

        // Create a single buffer with vertices for all gears
        let bufferSize = MemoryLayout<AAPLVertex>.stride * tempMeshes.reduce(into: 0) { partialResult, object in
            partialResult += object.count
        }

        guard let vbuffer = device.makeBuffer(length: bufferSize) else { fatalError() }
        vertexBuffer = vbuffer
        vertexBuffer.label = "Combined vertex buffer"

        // Copy each mesh's data into the vertex buffer
        let params = objectParameters.contents().bindMemory(to: AAPLObjectParameters.self, capacity: Int(AAPLNumObjects))

        var currentStartVertex = 0

        (0..<AAPLNumObjects).forEach { [vertexBuffer] objectIdx in
            let i = Int(objectIdx)
            // Store the mesh metadata in the params buffer
            params[i].numVertices = UInt32(tempMeshes[i].count)
            let meshSize = MemoryLayout<AAPLVertex>.stride * Int(tempMeshes[i].count)
            params[i].startVertex = UInt32(currentStartVertex)

            // Pack the current mesh data in the combined vertex buffer.
            let meshStartAddress = vertexBuffer.contents().advanced(by: currentStartVertex * MemoryLayout<AAPLVertex>.stride)
            memcpy(meshStartAddress, tempMeshes[i], meshSize)
            currentStartVertex += Int(tempMeshes[i].count)

            let gridPos: SIMD2<Float> = [Float(i % Int(AAPLGridWidth)), Float(i / Int(AAPLGridWidth))]
            params[i].position = gridPos * Float(AAPLObjecDistance)
            params[i].boundingRadius = Float(AAPLObjectSize) * 0.5
        }

        // Create buffers to contain dynamic shader data
        frameStateBuffer = (0..<maxFramesInFlight).map { i in
            guard let newBuffer = device.makeBuffer(length: MemoryLayout<AAPLFrameState>.stride,
                                                    options: .storageModeShared) else { fatalError() }
            newBuffer.label = "Frame state buffer \(i)"
            return newBuffer
        }

        let icbDescriptor = MTLIndirectCommandBufferDescriptor()

        // Indicate that the only draw commands will be standard (non-indexed) draw commands.
        icbDescriptor.commandTypes = .draw

        // Indicate that buffers will be set for each command IN the indirect command buffer
        icbDescriptor.inheritBuffers = false

        // Indicate that a max of 3 buffers will be set for each command.
        icbDescriptor.maxVertexBufferBindCount = 3
        icbDescriptor.maxFragmentBufferBindCount = 0

        icbDescriptor.inheritPipelineState = true

        guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDescriptor,
                                                         maxCommandCount: Int(AAPLNumObjects),
                                                         options: [.storageModePrivate])
        else { fatalError("Could not create indirect command buffer") }

        indirectCommandBuffer = icb
        indirectCommandBuffer.label = "Scene ICB"

        let argumentEncoder = gpuCommandEncodingKernel.makeArgumentEncoder(
            bufferIndex: Int(AAPLKernelBufferIndexCommandBufferContainer.rawValue))
        
        guard let icbBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: [.storageModeShared]) else {
            fatalError()
        }
        
        icbArgumentBuffer = icbBuffer
        icbArgumentBuffer.label = "ICB Argument Buffer"
        
        argumentEncoder.setArgumentBuffer(icbArgumentBuffer, offset: 0)
        argumentEncoder.setIndirectCommandBuffer(indirectCommandBuffer, index: Int(AAPLArgumentBufferIDCommandBuffer.rawValue))

        super.init()

        metalView.delegate = self
    }

    private static func makeGearMesh(numTeeth: Int, innerRatio: Float, toothWidth: Float, toothSlope: Float) -> [AAPLVertex] {
        assert(numTeeth >= 3, "Can only build a gear with at least 3 teeth")
        assert(toothWidth + 2 * toothSlope < 1.0, "Configuration of gear invalid")

        let angle = 2.0 * Float.pi / Float(numTeeth)
        let origin: SIMD2<Float> = .zero
        var meshVertices: [AAPLVertex] = Array(repeating: AAPLVertex(position: .zero, texcoord: .zero), count: numTeeth * 12)
        var vtx = 0

        for tooth in 0..<numTeeth {
            let fTooth = Float(tooth)
            // Calculate angles for tooth and groove
            let toothStartAngle = fTooth * angle
            let toothTip1Angle = (fTooth + toothSlope) * angle
            let toothTip2Angle = (fTooth + toothSlope + toothWidth) * angle
            let toothEndAngle = (fTooth + 2.0 * toothSlope + toothWidth) * angle
            let nextToothAngle = (fTooth + 1.0) * angle

            let groove1 = SIMD2<Float>(x: sinf(toothStartAngle) * innerRatio, y: cosf(toothStartAngle) * innerRatio)
            let tip1 = SIMD2<Float>(x: sinf(toothTip1Angle), y: cosf(toothTip1Angle))
            let tip2 = SIMD2<Float>(x: sinf(toothTip2Angle), y: cosf(toothTip2Angle))
            let groove2 = SIMD2<Float>(x: sinf(toothEndAngle) * innerRatio, y: cosf(toothEndAngle) * innerRatio)
            let nextGroove = SIMD2<Float>(x: sinf(nextToothAngle) * innerRatio, y: cosf(nextToothAngle) * innerRatio)

            let verts = [
                groove1, tip1, tip2,        // Right top triangle of tooth
                groove1, tip2, groove2,     // Left bottom triangle of tooth
                origin, groove1, groove2,   // Slice of circle from bottom of tooth to center of gear
                origin, groove2, nextGroove // Slice of circle from groove to center of gear
            ]

            verts.forEach { vertex in
                meshVertices[vtx] = AAPLVertex(position: vertex, texcoord: (vertex + 1.0) * 0.5)
                vtx += 1
            }
        }

        return meshVertices
    }

    private let rightBounds: Float  =  Float(AAPLObjecDistance) * Float(AAPLGridWidth) * 0.5
    private let leftBounds: Float   = -Float(AAPLObjecDistance) * Float(AAPLGridWidth) * 0.5
    private let upperBounds: Float  =  Float(AAPLObjecDistance) * Float(AAPLGridHeight) * 0.5
    private let lowerBounds: Float  = -Float(AAPLObjecDistance) * Float(AAPLGridHeight) * 0.5
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspectScale = [Float(size.height / size.width), 1.0]
    }

    private func updateState() {
        frameNumber += 1
        inFlightIndex = frameNumber % maxFramesInFlight

        movementSpeed = 0.15

        // Check if we've moved outside the grid boundaries and reverse direction if we have
        if  gridCenter.x < leftBounds ||
            gridCenter.x > rightBounds ||
            gridCenter.y < lowerBounds ||
            gridCenter.y > upperBounds {
            objectDirection = MovementDirection(rawValue: (objectDirection.rawValue + 2) % 4)!
        } else if frameNumber % 300 == 0 {
            objectDirection = MovementDirection(rawValue: Int.random(in: 0...3))!
        }

        switch objectDirection {
        case .right:
            gridCenter.x += movementSpeed
        case .up:
            gridCenter.y += movementSpeed
        case .left:
            gridCenter.x -= movementSpeed
        case .down:
            gridCenter.y -= movementSpeed
        }

        let gridDimensions: SIMD2<Float> = [ Float(AAPLGridWidth), Float(AAPLGridHeight) ]

        let frameState = frameStateBuffer[inFlightIndex].contents().bindMemory(to: AAPLFrameState.self, capacity: 1)
        frameState.pointee.aspectScale = aspectScale

        let viewOffset = Float(AAPLObjecDistance * 0.5) * (gridDimensions - 1)
        // Calculate the position of the center of the lower-left object
        frameState.pointee.translation = gridCenter - viewOffset
    }

    func draw(in view: MTKView) {
        inFlightSemaphore.wait()

        updateState()

        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
            inFlightSemaphore.signal()
        }

        commandBuffer.label = "Frame Command Buffer"

        // Encode command to reset the indirect command buffer
        let resetBlitEncoder = commandBuffer.makeBlitCommandEncoder()
        resetBlitEncoder?.label = "Reset ICB Blit Encoder"
        resetBlitEncoder?.resetCommandsInBuffer(indirectCommandBuffer, range: 0..<Int(AAPLNumObjects))
        resetBlitEncoder?.endEncoding()

        // Encode commands to determine visibility of objects using a compute kernel
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()
        computeEncoder?.setComputePipelineState(computePipelineState)
        computeEncoder?.setBuffer(frameStateBuffer[inFlightIndex], offset: 0, index: Int(AAPLKernelBufferIndexFrameState.rawValue))
        computeEncoder?.setBuffer(objectParameters, offset: 0, index: Int(AAPLKernelBufferIndexObjectParams.rawValue))
        computeEncoder?.setBuffer(vertexBuffer, offset: 0, index: Int(AAPLKernelBufferIndexVertices.rawValue))
        computeEncoder?.setBuffer(icbArgumentBuffer, offset: 0, index: Int(AAPLKernelBufferIndexCommandBufferContainer.rawValue))

        computeEncoder?.useResource(indirectCommandBuffer, usage: .write)

        let threadExecutionWidth = computePipelineState.threadExecutionWidth

        computeEncoder?.dispatchThreads(MTLSize(width: Int(AAPLNumObjects), height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: threadExecutionWidth, height: 1, depth: 1))

        computeEncoder?.endEncoding()

        // Encode command to optimize the indirect command buffer after encoding
        let optimizeBlitEncoder = commandBuffer.makeBlitCommandEncoder()
        optimizeBlitEncoder?.label = "Optimize ICB Blit Encoder"
        optimizeBlitEncoder?.optimizeIndirectCommandBuffer(indirectCommandBuffer, range: 0..<Int(AAPLNumObjects))
        optimizeBlitEncoder?.endEncoding()

        // If we've gotten a renderPassDescriptor we can render to the drawable, otherwise we'll skip
        // any rendering this frame because we have no drawable to draw to

        guard let renderEncoder =
                commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        renderEncoder.label = "Main Render Encoder"

        renderEncoder.setCullMode(.back)
        renderEncoder.setRenderPipelineState(renderPipelineState)

        renderEncoder.useResource(vertexBuffer, usage: .read, stages: .vertex)
        renderEncoder.useResource(objectParameters, usage: .read, stages: .vertex)
        renderEncoder.useResource(frameStateBuffer[inFlightIndex], usage: .read, stages: .vertex)

        // Draw everything in the indirect command buffer.
        renderEncoder.executeCommandsInBuffer(indirectCommandBuffer, range: 0..<Int(AAPLNumObjects))

        // We're done encoding commands
        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable, afterMinimumDuration: 0.016)
        }

        commandBuffer.commit()
    }
}
