import Foundation
import MetalKit

private let maxFramesInFlight = 3

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState

    private let inFlightSemaphore = DispatchSemaphore(value: maxFramesInFlight)

    private let vertexBuffer: [MTLBuffer]
    private let objectParameters: MTLBuffer
    private let frameStateBuffer: [MTLBuffer]
    private let indirectFrameStateBuffer: MTLBuffer
    private let indirectCommandBuffer: MTLIndirectCommandBuffer

    var inFlightIndex = 0
    var frameNumber = 0

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

        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        vertexBuffer = (0..<AAPLNumObjects).map { i in
            // Choose parameters to generate a mesh for this object so that each mesh is unique
            // and looks different than the mesh it is next to in the grid
            let numTeeth = i < 8 ? i + 3 : i * 3

            // Create a vertex buffer, and initialize it with a unique 2D gear mesh
            let buffer = Renderer.makeGearMesh(device: device, numTeeth: Int(numTeeth))
            buffer.label = "Object \(i) Buffer"
            return buffer
        }

        // Create and fill array containing parameters for each object
        let objectParameterArraySize = Int(AAPLNumObjects) * MemoryLayout<AAPLObjectParameters>.stride

        guard let paramBuffer = device.makeBuffer(length: objectParameterArraySize) else { fatalError() }
        objectParameters = paramBuffer
        objectParameters.label = "Object Parameters Array"

        let params = objectParameters.contents().bindMemory(to: AAPLObjectParameters.self, capacity: Int(AAPLNumObjects))
        let offset = Float(AAPLObjectDistance / 2.0) * (Renderer.gridDimensions - Float(1.0))
        let gridWidth = Int(AAPLGridWidth)

        (0..<Int(AAPLNumObjects)).forEach { i in
            // Calculate position of each object such that each occupies a space in a grid
            let gridPos: SIMD2<Float> = [Float(i % gridWidth), Float(i / gridWidth)]
            let position: SIMD2<Float> = -offset + gridPos * Float(AAPLObjectDistance)
            params[i].position = position
        }

        frameStateBuffer = (0..<maxFramesInFlight).map { i in
            guard let newBuffer = device.makeBuffer(length: MemoryLayout<AAPLFrameState>.stride,
                                                    options: .storageModeShared) else { fatalError() }
            newBuffer.label = "Frame state buffer \(i)"
            return newBuffer
        }

        // When encoding commands with the CPU, the app sets this indirect frame state buffer
        // dynamically in the indirect command buffer. Each frame data will be blit from the
        // frameStateBuffer that has just been updated by the CPU to this buffer. This allows
        // a synchronous update of values set by the CPU.
        guard let indirectBuffer = device.makeBuffer(length: MemoryLayout<AAPLFrameState>.stride,
                                                     options: .storageModePrivate) else { fatalError() }

        indirectFrameStateBuffer = indirectBuffer
        indirectFrameStateBuffer.label = "Indirect frame state buffer"

        let icbDescriptor = MTLIndirectCommandBufferDescriptor()

        // Indicate that the only draw commands will be standard (non-indexed) draw commands.
        icbDescriptor.commandTypes = .draw

        // Indicate that buffers will be set for each command IN the indirect command buffer
        icbDescriptor.inheritBuffers = false

        // Indicate that a max of 3 buffers will be set for each command.
        icbDescriptor.maxVertexBufferBindCount = 3
        icbDescriptor.maxFragmentBufferBindCount = 0

        icbDescriptor.inheritPipelineState = true

        guard let icb = device.makeIndirectCommandBuffer(descriptor: icbDescriptor, maxCommandCount: Int(AAPLNumObjects))
        else { fatalError("Could not create indirect command buffer") }

        indirectCommandBuffer = icb
        indirectCommandBuffer.label = "Scene ICB"

        super.init()

        // Encode a draw command for each object drawn in the indirect command buffer.
        (0..<AAPLNumObjects).forEach { i in
            let objIndex = Int(i)
            let icbCommand = indirectCommandBuffer.indirectRenderCommandAt(objIndex)

            icbCommand.setVertexBuffer(vertexBuffer[objIndex], offset: 0, at: Int(AAPLVertexBufferIndexVertices.rawValue))
            icbCommand.setVertexBuffer(indirectFrameStateBuffer, offset: 0, at: Int(AAPLVertexBufferIndexFrameState.rawValue))
            icbCommand.setVertexBuffer(objectParameters, offset: 0, at: Int(AAPLVertexBufferIndexObjectParams.rawValue))

            let vertexCount = vertexBuffer[objIndex].length / MemoryLayout<AAPLVertex>.stride

            icbCommand.drawPrimitives(.triangle, vertexStart: 0, vertexCount: vertexCount,
                                      instanceCount: 1, baseInstance: objIndex)
        }

        metalView.delegate = self
    }

    private static func makeGearMesh(device: MTLDevice, numTeeth: Int) -> MTLBuffer {
        assert(numTeeth >= 3, "Can only build a gear with at least 3 teeth")

        let innerRatio = Float(0.8)
        let toothWidth = Float(0.25)
        let toothSlope = Float(0.2)

        let numVertices = numTeeth * 12
        let bufferSize = MemoryLayout<AAPLVertex>.stride * numVertices
        guard let metalBuffer = device.makeBuffer(length: bufferSize) else { fatalError() }
        metalBuffer.label = "\(numTeeth) Toothed Cog Vertices"

        let meshVertices = metalBuffer.contents().bindMemory(to: AAPLVertex.self, capacity: numVertices)

        let angle = 2.0 * Float.pi / Float(numTeeth)
        let origin: SIMD2<Float> = [0, 0]
        var vtx = 0

        for tooth in 0..<numTeeth {
            let dTooth = Float(tooth)
            // Calculate angles for tooth and groove
            let toothStartAngle = dTooth * angle
            let toothTip1Angle = (dTooth + toothSlope) * angle
            let toothTip2Angle = (dTooth + toothSlope + toothWidth) * angle
            let toothEndAngle = (dTooth + 2.0 * toothSlope + toothWidth) * angle
            let nextToothAngle = (dTooth + 1.0) * angle

            let groove1: SIMD2<Float> = [ sin(toothStartAngle) * innerRatio, cos(toothStartAngle) * innerRatio]
            let tip1: SIMD2<Float> = [ sin(toothTip1Angle), cos(toothTip1Angle)]
            let tip2: SIMD2<Float> = [ sin(toothTip2Angle), cos(toothTip2Angle)]
            let groove2: SIMD2<Float> = [ sin(toothEndAngle) * innerRatio, cos(toothEndAngle) * innerRatio]
            let nextGroove: SIMD2<Float> = [ sin(nextToothAngle) * innerRatio, cos(nextToothAngle) * innerRatio]

            let verts = [
                groove1, tip1, tip2,        // Right top triangle of tooth
                groove1, tip2, groove2,     // Left bottom triangle of tooth
                origin, groove1, groove2,   // Slice of circle from bottom of tooth to center of gear
                origin, groove2, nextGroove // Slice of circle from groove to center of gear
            ]

            verts.forEach { vertex in
                meshVertices[vtx].position = vertex
                meshVertices[vtx].texcoord = (vertex + 1.0) * 0.5
                vtx += 1
            }
        }

        return metalBuffer
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspectScale = [Float(size.height / size.width), 1.0]
    }

    private func updateState() {
        frameNumber += 1
        inFlightIndex = frameNumber % maxFramesInFlight
        let frameState = frameStateBuffer[inFlightIndex].contents().bindMemory(to: AAPLFrameState.self, capacity: 1)
        frameState.pointee.aspectScale = aspectScale
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

        // Encode blit commands to update the buffer holding the frame state.
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()

        blitEncoder?.copy(from: frameStateBuffer[inFlightIndex], sourceOffset: 0,
                          to: indirectFrameStateBuffer, destinationOffset: 0,
                          size: indirectFrameStateBuffer.length)

        blitEncoder?.endEncoding()

        guard let renderEncoder =
                commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        renderEncoder.label = "Main Render Encoder"

        renderEncoder.setCullMode(.back)
        renderEncoder.setRenderPipelineState(pipelineState)

        // Make a useResource call for each buffer needed by the indirect command buffer.
        for i in 0..<Int(AAPLNumObjects) {
            renderEncoder.useResource(vertexBuffer[i], usage: .read, stages: .vertex)
        }

        renderEncoder.useResource(objectParameters, usage: .read, stages: .vertex)
        renderEncoder.useResource(indirectFrameStateBuffer, usage: .read, stages: .vertex)

        // Draw everything in the indirect command buffer.
        renderEncoder.executeCommandsInBuffer(indirectCommandBuffer, range: 0..<Int(AAPLNumObjects))

        // We're done encoding commands
        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
