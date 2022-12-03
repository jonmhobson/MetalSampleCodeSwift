import Foundation
import MetalKit

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let pipelineState: MTLRenderPipelineState

    private let vertexBuffer: MTLBuffer
    private let textures: [MTLTexture]
    private let dataBuffers: [MTLBuffer]
    private let heap: MTLHeap
    private let fragmentShaderArgumentBuffer: MTLBuffer

    private var viewport: MTLViewport = .init()

    static let vertexData: [Vertex] = [
        .init(position: [+0.75, -0.75], texCoord: [1, 0]),
        .init(position: [-0.75, -0.75], texCoord: [0, 0]),
        .init(position: [-0.75, +0.75], texCoord: [0, 1]),
        .init(position: [+0.75, -0.75], texCoord: [1, 0]),
        .init(position: [-0.75, +0.75], texCoord: [0, 1]),
        .init(position: [+0.75, +0.75], texCoord: [1, 1]),
    ]

    static func loadTextures(device: MTLDevice) -> [MTLTexture] {
        let textureLoader = MTKTextureLoader(device: device)

        let textures: [MTLTexture] = (0..<Int(NumTextureArguments.rawValue)).map {
            let textureName = "Texture\($0)"
            do {
                let texture = try textureLoader.newTexture(name: textureName, scaleFactor: 1.0, bundle: nil)
                texture.label = textureName
                return texture
            } catch {
                fatalError(error.localizedDescription)
            }
        }

        return textures
    }

    static func createBuffers(device: MTLDevice) -> [MTLBuffer] {
        let buffers: [MTLBuffer] = (0..<Int(NumBufferArguments.rawValue)).map { (i: Int) -> MTLBuffer in
            let elementCount = Int.random(in: 128..<384)

            guard let buffer = device.makeBuffer(length: MemoryLayout<Float>.stride * elementCount, options: .storageModeShared) else {
                fatalError("Failed to create buffer")
            }

            buffer.label = "DataBuffer\(i)"

            let elements = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)

            for k in 0..<elementCount {
                let point: Float = (Float(k) * 2.0 * .pi) / Float(elementCount)
                elements[k] = sinf(point * Float(i)) * 0.5 + 0.5
            }

            return buffer
        }

        return buffers
    }

    static func descriptorFromTexture(texture: MTLTexture, storageMode: MTLStorageMode) -> MTLTextureDescriptor {
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

    static func createHeap(device: MTLDevice, textures: [MTLTexture], buffers: [MTLBuffer]) -> MTLHeap {
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.storageMode = .private
        heapDescriptor.size = 0

        // Build a descriptor for each texture and calculate the size required to store all textures in the heap
        for texture in textures {
            let descriptor = Self.descriptorFromTexture(texture: texture, storageMode: heapDescriptor.storageMode)

            // Determine the size required for the heap for the given descriptor
            var sizeAndAlign = device.heapTextureSizeAndAlign(descriptor: descriptor)

            // Align the size so that more resources will fit in the heap after this texture
            sizeAndAlign.size += (sizeAndAlign.size & (sizeAndAlign.align - 1)) + sizeAndAlign.align

            // Accumulate the size required to store this texture in the heap
            heapDescriptor.size += sizeAndAlign.size
        }

        // Calculate the size required to store all buffers in the heap
        for buffer in buffers {
            // Determine the size required for the heap for the given buffer size
            var sizeAndAlign = device.heapBufferSizeAndAlign(length: buffer.length,
                                                             options: .storageModePrivate)

            // Align the size so that more resources will fit in the heap after this buffer
            sizeAndAlign.size += (sizeAndAlign.size & (sizeAndAlign.align - 1)) + sizeAndAlign.align

            // Accumulate the size required to store this buffer in the heap
            heapDescriptor.size += sizeAndAlign.size
        }

        guard let heap = device.makeHeap(descriptor: heapDescriptor) else { fatalError() }
        return heap
    }

    static func moveResources(to heap: MTLHeap, commandQueue: MTLCommandQueue,
                              textures: inout [MTLTexture], buffers: inout [MTLBuffer]) {
        // Create a command buffer and blit encoder to copy data from the existing resources to
        // the new resources created from the heap
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { fatalError() }

        commandBuffer.label = "Heap Copy Command Buffer"
        blitEncoder.label = "Heap Transfer Blit Encoder"

        // Create new textures from the heap and copy the contents of the existing textures to the new textures
        for (i, texture) in textures.enumerated() {
            let descriptor = Self.descriptorFromTexture(texture: texture, storageMode: heap.storageMode)

            // Create a texture from the heap
            guard let heapTexture = heap.makeTexture(descriptor: descriptor) else { fatalError("Failed to create heap texture") }
            heapTexture.label = texture.label

            blitEncoder.pushDebugGroup("\(heapTexture.label!) Blits")

            // Blit every slice of every level from the existing texture to the new texture
            var region = MTLRegionMake2D(0, 0, texture.width, texture.height)
            for level in 0..<texture.mipmapLevelCount {
                blitEncoder.pushDebugGroup("Level \(level) Blit")
                for slice in 0..<texture.arrayLength {
                    blitEncoder.copy(
                        from: texture,
                        sourceSlice: slice,
                        sourceLevel: level,
                        sourceOrigin: region.origin,
                        sourceSize: region.size,
                        to: heapTexture,
                        destinationSlice: slice,
                        destinationLevel: level,
                        destinationOrigin: region.origin
                    )
                }
                region.size.width = max(1, region.size.width / 2)
                region.size.height = max(1, region.size.height / 2)

                blitEncoder.popDebugGroup()
            }
            blitEncoder.popDebugGroup()

            // Replace the existing texture with the new texture
            textures[i] = heapTexture
        }

        // Create new buffers from the heap and copy the contents of existing buffers to the
        // new buffers
        for (i, buffer) in buffers.enumerated() {
            // Create a buffer from the heap
            guard let heapBuffer = heap.makeBuffer(length: buffer.length, options: .storageModePrivate) else {
                fatalError("Failed to create buffer on heap")
            }

            heapBuffer.label = buffer.label

            // Blit contents of the original buffer to the new buffer
            blitEncoder.copy(from: buffer, sourceOffset: 0, to: heapBuffer, destinationOffset: 0, size: heapBuffer.length)

            // Replace the existing buffer with the new buffer
            buffers[i] = heapBuffer
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
        metalView.clearColor = MTLClearColor(red: 0, green: 0.5, blue: 0.5, alpha: 1)

        guard let vertexBuffer = device.makeBuffer(
            bytes: Self.vertexData,
            length: MemoryLayout<Vertex>.stride * Self.vertexData.count,
            options: .storageModeShared
        ) else { fatalError("Failed to create vertex buffer")}

        self.vertexBuffer = vertexBuffer
        self.vertexBuffer.label = "Vertices"

        var textures = Self.loadTextures(device: device)
        var dataBuffers = Self.createBuffers(device: device)

        self.heap = Self.createHeap(device: device, textures: textures, buffers: dataBuffers)

        Self.moveResources(to: self.heap,
                           commandQueue: self.commandQueue,
                           textures: &textures,
                           buffers: &dataBuffers)

        self.textures = textures
        self.dataBuffers = dataBuffers

        // Create our render pipeline
        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else {
            fatalError("Failed to create shaders")
        }

        // Create a pipeline state object
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Argument Buffer Example"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state, error \(error.localizedDescription)")
            }
        }()

        let argumentEncoder = fragmentFunction.makeArgumentEncoder(bufferIndex: Int(FragmentBufferIndexArguments.rawValue))
        let argumentBufferLength = argumentEncoder.encodedLength

        guard let argumentBuffer = device.makeBuffer(length: argumentBufferLength) else { fatalError() }

        self.fragmentShaderArgumentBuffer = argumentBuffer
        self.fragmentShaderArgumentBuffer.label = "Argument Buffer Fragment Shader"

        argumentEncoder.setArgumentBuffer(fragmentShaderArgumentBuffer, offset: 0)

        for (i, texture) in textures.enumerated() {
            argumentEncoder.setTexture(texture, index: Int(ArgumentBufferIDExampleTextures.rawValue) + i)
        }

        for (i, buffer) in dataBuffers.enumerated() {
            argumentEncoder.setBuffer(buffer, offset: 0, index: Int(ArgumentBufferIDExampleBuffers.rawValue) + i)

            let elementCountAddress = argumentEncoder.constantData(at: Int(ArgumentBufferIDExampleConstants.rawValue) + i)
            let pointer = elementCountAddress.bindMemory(to: UInt32.self, capacity: 1)
            pointer.pointee = UInt32(buffer.length / 4)
        }

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Calculate a viewport so that it's always square and in the middle of the drawable

        if size.width < size.height {
            viewport.originX = 0
            viewport.originY = (size.height - size.width) * 0.5
            viewport.width = size.width
            viewport.height = size.width
        } else {
            viewport.originX = (size.width - size.height) * 0.5
            viewport.originY = 0
            viewport.width = size.height
            viewport.height = size.height
        }

        viewport.zfar = 1.0
        viewport.znear = -1.0
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        commandBuffer.label = "Per Frame Commands"
        renderEncoder.label = "Per Frame Rendering"

        renderEncoder.setViewport(viewport)

        renderEncoder.useHeap(heap, stages: .fragment)
        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setVertexBuffer(vertexBuffer,
                                      offset: 0,
                                      index: Int(VertexBufferIndexVertices.rawValue))
        renderEncoder.setFragmentBuffer(fragmentShaderArgumentBuffer,
                                        offset: 0,
                                        index: Int(FragmentBufferIndexArguments.rawValue))

        renderEncoder.drawPrimitives(type: .triangle,
                                     vertexStart: 0, vertexCount: 6)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
