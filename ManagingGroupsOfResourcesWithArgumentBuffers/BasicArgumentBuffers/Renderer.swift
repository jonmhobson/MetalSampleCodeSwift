import Foundation
import MetalKit

struct FragmentShaderArguments {
    var exampleTexture: MTLResourceID
    var exampleSampler: MTLResourceID
    var exampleBuffer: UnsafeRawPointer
    var exampleConstant: UInt32
}

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let vertexBuffer: MTLBuffer
    private let texture: MTLTexture
    private let sampler: MTLSamplerState
    private let indirectBuffer: MTLBuffer
    private let pipelineState: MTLRenderPipelineState
    private let fragmentShaderArgumentBuffer: MTLBuffer

    private var viewport = MTLViewport(originX: 0, originY: 0, width: 0, height: 0, znear: 0, zfar: 0)

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 0.5, alpha: 1.0)
        metalView.colorPixelFormat = .bgra8Unorm_srgb

        // Create a vertex buffer and initialize it with the generics array.
        let vertexData = [
            AAPLVertex(position: [ 0.75, -0.75], texCoord: [1.0, 0.0], color: [0, 1, 0, 1]),
            AAPLVertex(position: [-0.75, -0.75], texCoord: [0.0, 0.0], color: [1, 1, 1, 1]),
            AAPLVertex(position: [-0.75,  0.75], texCoord: [0.0, 1.0], color: [0, 0, 1, 1]),
            AAPLVertex(position: [ 0.75, -0.75], texCoord: [1.0, 0.0], color: [0, 1, 0, 1]),
            AAPLVertex(position: [-0.75,  0.75], texCoord: [0.0, 1.0], color: [0, 0, 1, 1]),
            AAPLVertex(position: [ 0.75,  0.75], texCoord: [1.0, 1.0], color: [1, 1, 1, 1])
        ]

        guard let vertexBuffer = device.makeBuffer(bytes: vertexData, length: vertexData.count * MemoryLayout<AAPLVertex>.stride, options: [.storageModeShared]) else { fatalError() }

        self.vertexBuffer = vertexBuffer
        self.vertexBuffer.label = "Vertices"

        // Create texture to apply to the quad.
        let textureLoader = MTKTextureLoader(device: device)
        self.texture = {
            do {
                return try textureLoader.newTexture(name: "Text", scaleFactor: 1.0, bundle: nil)
            } catch {
                fatalError("Could not load foregroundTexture: \(error)")
            }
        }()
        self.texture.label = "Text"

        // Create a sampler to use for texturing
        self.sampler = {
            let samplerDesc = MTLSamplerDescriptor()
            samplerDesc.minFilter = .linear
            samplerDesc.magFilter = .linear
            samplerDesc.mipFilter = .notMipmapped
            samplerDesc.normalizedCoordinates = true
            samplerDesc.supportArgumentBuffers = true

            guard let sampler = device.makeSamplerState(descriptor: samplerDesc) else { fatalError() }
            return sampler
        }()

        let bufferElements = 256

        // Create buffers for making a pattern on the quad.
        self.indirectBuffer = {
            guard let indirectBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * bufferElements) else {
                fatalError()
            }

            let patternArray = indirectBuffer.contents().bindMemory(to: Float.self, capacity: bufferElements)
            for i in 0..<bufferElements {
                patternArray[i] = ((i % 24) < 3) ? 1.0 : 0.0
            }

            indirectBuffer.label = "Indirect Buffer"
            return indirectBuffer
        }()

        // Create the render pipeline state.
        self.pipelineState = {
            guard let defaultLibrary = device.makeDefaultLibrary(),
                  let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
                  let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

            let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
            pipelineStateDescriptor.label = "Argument Buffer Example"
            pipelineStateDescriptor.vertexFunction = vertexFunction
            pipelineStateDescriptor.fragmentFunction = fragmentFunction
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat

            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state")
            }
        }()

        // Create the argument buffer
        self.fragmentShaderArgumentBuffer = {
            assert(device.argumentBuffersSupport != .tier1, "Metal 3 argument buffers are only supported on Tier2 devices")
            let argumentBufferLength = MemoryLayout<FragmentShaderArguments>.stride
            guard let argumentBuffer = device.makeBuffer(length: argumentBufferLength) else { fatalError() }
            return argumentBuffer
        }()

        let argumentStructure = fragmentShaderArgumentBuffer.contents().bindMemory(to: FragmentShaderArguments.self, capacity: 1)
        argumentStructure.pointee.exampleTexture = texture.gpuResourceID
        argumentStructure.pointee.exampleBuffer = UnsafeRawPointer(bitPattern: UInt(indirectBuffer.gpuAddress))!
        argumentStructure.pointee.exampleSampler = sampler.gpuResourceID
        argumentStructure.pointee.exampleConstant = UInt32(bufferElements)

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Calculate a viewport so that it's always square and in the middle of the drawable.
        if size.width < size.height {
            viewport.originX = 0.0
            viewport.originY = (size.height - size.width) * 0.5
            viewport.height = size.width
            viewport.width = size.width
        } else {
            viewport.originX = (size.width - size.height) * 0.5
            viewport.originY = 0.0
            viewport.height = size.height
            viewport.width = size.height
        }

        viewport.zfar = 1.0
        viewport.znear = -1.0
    }

    func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        commandBuffer.label = "MyCommand"

        // Create a render command encoder to render with.
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        renderEncoder.label = "MyRenderEncoder"

        renderEncoder.setViewport(viewport)

        // Indicate to Metal that the GPU accesses these resources, so they need
        // to map to the GPU's address space.
        renderEncoder.useResource(texture, usage: .read, stages: .fragment)
        renderEncoder.useResource(indirectBuffer, usage: .read, stages: .fragment)

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: Int(AAPLVertexBufferIndexVertices.rawValue))
        renderEncoder.setFragmentBuffer(fragmentShaderArgumentBuffer, offset: 0, index: Int(AAPLFragmentBufferIndexArguments.rawValue))

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
