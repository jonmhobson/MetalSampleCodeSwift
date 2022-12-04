import Foundation
import MetalKit

private let maxBuffersInFlight = 3

struct InstanceArguments {
    let position: vector_float2
    let leftTexture: MTLResourceID
    let rightTexture: MTLResourceID
}

final class Renderer: NSObject {
    private let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Compute pipeline which updates instances and encodes instance parameters
    private let computePipeline: MTLComputePipelineState
    // The Metal buffer storing vertex data
    private let vertexBuffer: MTLBuffer
    // The Metal buffers storing per frame uniform data
    private let frameStateBuffer: [MTLBuffer]
    // Index into frameStateBuffer to use for the current frame
    private var inFlightIndex: Int = 0
    private let renderPipeline: MTLRenderPipelineState

    // Metal texture objects to be referenced via an argument buffer
    private let textures: [MTLTexture]
    // Buffer with each texture encoded into it
    private let sourceTextures: MTLBuffer
    // Buffer with parameters for each instance. Provides location and textures for quad instances.
    // Written by a compute kernel. Read by a render pipeline.
    private let instanceParameters: MTLBuffer
    // Resource heap to contain all resources encoded in our argument buffer
    private let heap: MTLHeap

    // Compute kernel dispatch parameters
    private var threadgroupSize: MTLSize
    private var threadgroupCount: MTLSize

    private var blendTheta: Float = 0.0
    private var textureIndexOffset: UInt32 = 0
    private var quadScale: vector_float2 = .zero

    static func descriptorFrom(texture: MTLTexture, storageMode: MTLStorageMode) -> MTLTextureDescriptor {
        let descriptor = MTLTextureDescriptor()

        descriptor.textureType = texture.textureType
        descriptor.pixelFormat = texture.pixelFormat
        descriptor.width = texture.width
        descriptor.height = texture.height
        descriptor.depth = texture.depth
        descriptor.mipmapLevelCount = texture.mipmapLevelCount
        descriptor.arrayLength = texture.arrayLength
        descriptor.sampleCount = texture.sampleCount
        descriptor.storageMode = storageMode

        return descriptor
    }

    static func moveResourcesTo(heap: MTLHeap, commandQueue: MTLCommandQueue, textures: inout [MTLTexture]) {
        // Create a command buffer and blit encoder to upload data from original resources to newly created
        // resources from the heap
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { fatalError() }

        commandBuffer.label = "Heap upload command buffer"
        blitEncoder.label = "Heap transfer blit encoder"

        for (i, texture) in textures.enumerated() {
            // Create descriptor using the texture's properties
            let descriptor = Self.descriptorFrom(texture: texture, storageMode: heap.storageMode)

            // Create a texture from the heap
            guard let heapTexture = heap.makeTexture(descriptor: descriptor) else { fatalError() }
            heapTexture.label = texture.label

            blitEncoder.pushDebugGroup("\(heapTexture.label ?? "") Blits")

            // Blit every slice of every level from the original texture to the texture ccreated from the heap
            var region = MTLRegionMake2D(0, 0, texture.width, texture.height)

            for level in 0..<texture.mipmapLevelCount {
                blitEncoder.pushDebugGroup("Level \(level) blit")

                for slice in 0..<texture.arrayLength {
                    blitEncoder.copy(from: texture,
                                     sourceSlice: slice,
                                     sourceLevel: level,
                                     sourceOrigin: region.origin,
                                     sourceSize: region.size,
                                     to: heapTexture,
                                     destinationSlice: slice,
                                     destinationLevel: level,
                                     destinationOrigin: region.origin)
                }

                region.size.width = max(1, region.size.width / 2)
                region.size.height = max(1, region.size.height / 2)

                blitEncoder.popDebugGroup()
            }

            blitEncoder.popDebugGroup()
            textures[i] = heapTexture
        }

        blitEncoder.endEncoding()
        commandBuffer.commit()
    }

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 0.5, alpha: 1.0)

        let quadSize = Float(QuadSize)

        // Create a vertex buffer and initialize it with the generics array.
        let vertexData: [Vertex] = [
            .init(position: [quadSize   , 0         ], texCoord: [1.0, 0.0]),
            .init(position: [0          , 0         ], texCoord: [0.0, 0.0]),
            .init(position: [0          , quadSize  ], texCoord: [0.0, 1.0]),
            .init(position: [quadSize   , 0         ], texCoord: [1.0, 0.0]),
            .init(position: [0          , quadSize  ], texCoord: [0.0, 1.0]),
            .init(position: [quadSize   , quadSize  ], texCoord: [1.0, 1.0])
        ]

        guard let vertexBuffer = device.makeBuffer(
            bytes: vertexData,
            length: vertexData.count * MemoryLayout<Vertex>.stride,
            options: [.storageModeShared]
        ) else { fatalError() }
        self.vertexBuffer = vertexBuffer
        self.vertexBuffer.label = "Vertices"

        // MARK: loadResources
        let textureLoader = MTKTextureLoader(device: device)
        var textures = (0..<NumTextures).map { i in
            let textureName = "Texture\(i)"
            do {
                let texture = try textureLoader.newTexture(name: textureName, scaleFactor: 1.0, bundle: nil)
                texture.label = textureName
                return texture
            } catch {
                fatalError("Could not load texture with name \(textureName): \(error.localizedDescription)")
            }
        }

        // MARK: createHeap
        self.heap = {
            let heapDescriptor = MTLHeapDescriptor()
            heapDescriptor.storageMode = .private
            heapDescriptor.size = 0

            for texture in textures {
                let descriptor = Self.descriptorFrom(texture: texture, storageMode: heapDescriptor.storageMode)
                // Determine size of needed from the heap given the descriptor
                var sizeAndAlign = device.heapTextureSizeAndAlign(descriptor: descriptor)
                // Align the size so that more resources will fit after this texture
                sizeAndAlign.size += (sizeAndAlign.size & (sizeAndAlign.align - 1)) + sizeAndAlign.align
                // Accumulate the size required for the heap to hold this texture
                heapDescriptor.size += sizeAndAlign.size
            }

            guard let heap = device.makeHeap(descriptor: heapDescriptor) else { fatalError() }
            heap.label = "Texture heap"
            return heap
        }()

        Self.moveResourcesTo(heap: heap, commandQueue: commandQueue, textures: &textures)
        self.textures = textures

        guard let defaultLibrary = device.makeDefaultLibrary() else { fatalError() }
        guard let computeFunction = defaultLibrary.makeFunction(name: "updateInstances") else {
            fatalError("Could not find compute function")
        }

        self.computePipeline = {
            do {
                let pipeline = try device.makeComputePipelineState(function: computeFunction)
                return pipeline
            } catch {
                fatalError("Failed to create compute pipeline state, error \(error.localizedDescription)")
            }
        }()

        threadgroupSize = MTLSize(width: 16, height: 1, depth: 1)
        threadgroupCount = MTLSize(width: 1, height: 1, depth: 1)
        threadgroupCount.width = max((2 * Int(NumInstances) - 1) / threadgroupSize.width, 1)

        // Create buffers for arguments
        // Calculate the size of the array of texture arguments necessary to fit all textures in the buffer.
        let textureArgumentArrayLength = MemoryLayout<MTLResourceID>.stride * Int(NumTextures)

        // Create a buffer that will hold the arguments for all textures
        sourceTextures = device.makeBuffer(length: textureArgumentArrayLength)!
        sourceTextures.label = "Texture list"

        let texPtr = sourceTextures.contents().bindMemory(to: MTLResourceID.self, capacity: textures.count)

        // Encode input arguments for our compute kernel
        for (i, texture) in textures.enumerated() {
            texPtr[i] = texture.gpuResourceID
        }

        // Create a buffer used for outputs from the compute kernel and inputs
        // for the render pipeline.

        // The encodedLength represents the size of the structure used to define the argument
        // buffer. Each instance needs its own structure, so we multiply encodedLength by the
        // number of instances so that we create a buffer which can hold data for each instance
        // rendered.
        let instanceParameterLength = MemoryLayout<InstanceArguments>.stride * Int(NumInstances)

        self.instanceParameters = device.makeBuffer(length: instanceParameterLength)!
        self.instanceParameters.label = "Instance parameters array"

        // Create render pipeline and objects used in the render pass
        guard let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else {
            fatalError("Could not load shaders")
        }

        // Create a pipeline state object
        renderPipeline = {
            let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
            pipelineStateDescriptor.label = "Argument buffer pipeline"
            pipelineStateDescriptor.vertexFunction = vertexFunction
            pipelineStateDescriptor.fragmentFunction = fragmentFunction
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create render pipeline state, error \(error.localizedDescription)")
            }
        }()

        frameStateBuffer = (0..<maxBuffersInFlight).map { i in
            guard let buffer = device.makeBuffer(length: MemoryLayout<FrameState>.stride, options: .storageModeShared) else {
                fatalError()
            }
            buffer.label = "FrameDataBuffer \(i)"
            return buffer
        }

        super.init()

        metalView.delegate = self
    }

    func updateState() {
        let frameState = frameStateBuffer[inFlightIndex].contents().bindMemory(to: FrameState.self, capacity: 1)

        blendTheta += 0.025
        frameState.pointee.quadScale = quadScale

        let gridHeight: Float = (Float(NumInstances+1) / Float(GridWidth))

        let halfGridDimensions = vector_float2(0.5 * Float(GridWidth), 0.5 * gridHeight)
        frameState.pointee.offset.x = Float(QuadSpacing) * quadScale.x * (halfGridDimensions.x - 1)
        frameState.pointee.offset.y = Float(QuadSpacing) * quadScale.y * -halfGridDimensions.y

        // Calculate a blend factor between 0 and 1. Using a sinusoidal equation makes the transition
        // period quicker and a single unblended image on the quad for longer (i.e. we move
        // quickly through a blend factor of 0.5 where the two textures are presented equally)
        frameState.pointee.slideFactor = (cosf(blendTheta + .pi) + 1.0) / 2.0
        frameState.pointee.textureIndexOffset = textureIndexOffset

        if blendTheta >= .pi {
            blendTheta = 0.0
            textureIndexOffset += 1
        }
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Calculate scale for quads so that they are always square when working with the default
        // viewport and sending down clip space coordinates
        if size.width < size.height {
            quadScale.x = 1.0
            quadScale.y = Float(size.width / size.height)
        } else {
            quadScale.x = Float(size.height / size.width)
            quadScale.y = 1.0
        }
    }

    func draw(in view: MTKView) {
        inFlightSemaphore.wait()
        inFlightIndex = (inFlightIndex + 1) % maxBuffersInFlight

        self.updateState()

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Per Frame Commands"

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Per Frame Compute Commands"

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(sourceTextures, offset: 0, index: Int(ComputeBufferIndexSourceTextures.rawValue))
        computeEncoder.setBuffer(frameStateBuffer[inFlightIndex], offset: 0, index: Int(ComputeBufferIndexFrameState.rawValue))
        computeEncoder.setBuffer(instanceParameters, offset: 0, index: Int(ComputeBufferIndexInstanceParams.rawValue))
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()

        // Obtain a renderPassDescriptor generated from the view's drawable textures
        if let renderPassDescriptor = view.currentRenderPassDescriptor,
           let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {

            renderEncoder.label = "Per Frame Rendering"

            // Make a single useHeap call instead of one useResource call per texture and per buffer,
            // since all buffers have been moved to memory in a heap
            renderEncoder.useHeap(heap, stages: .fragment)
            renderEncoder.setRenderPipelineState(renderPipeline)

            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: Int(VertexBufferIndexVertices.rawValue))
            renderEncoder.setVertexBuffer(frameStateBuffer[inFlightIndex], offset: 0, index: Int(VertexBufferIndexFrameState.rawValue))
            renderEncoder.setVertexBuffer(instanceParameters, offset: 0, index: Int(VertexBufferIndexInstanceParams.rawValue))

            renderEncoder.setFragmentBuffer(instanceParameters, offset: 0, index: Int(FragmentBufferIndexInstanceParams.rawValue))
            renderEncoder.setFragmentBuffer(frameStateBuffer[inFlightIndex], offset: 0, index: Int(FragmentBufferIndexFrameState.rawValue))

            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: Int(NumInstances))

            renderEncoder.endEncoding()

            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
        }

        commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
            inFlightSemaphore.signal()
        }

        commandBuffer.commit()
    }
}
