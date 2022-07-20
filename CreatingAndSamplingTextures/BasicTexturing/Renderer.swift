import Foundation
import MetalKit

final class Renderer: NSObject {
    private let vertices: MTLBuffer
    private let numVertices: Int
    private let texture: MTLTexture

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState

    private var viewportSize: simd_uint2 = [0, 0]

    private static let triangleVertices: [AAPLVertex] = [
        AAPLVertex(position: [ 250, -250], textureCoordinate: [1, 1]),
        AAPLVertex(position: [-250, -250], textureCoordinate: [0, 1]),
        AAPLVertex(position: [-250,  250], textureCoordinate: [0, 0]),

        AAPLVertex(position: [ 250, -250], textureCoordinate: [1, 1]),
        AAPLVertex(position: [-250,  250], textureCoordinate: [0, 0]),
        AAPLVertex(position: [ 250,  250], textureCoordinate: [1, 0])
    ]

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue
        let textureLoader = MTKTextureLoader(device: device)

        // The original sample loads a TGA file manually. I think that distracts from the main point of the sample so
        // instead here we just load a texture from the Assets catalog using the MetalKit texture loader
        self.texture = {
            do {
                return try textureLoader.newTexture(name: "Image", scaleFactor: metalView.window?.backingScaleFactor ?? 1.0, bundle: nil)
            } catch {
                fatalError("Could not load texture. Error: \(error)")
            }
        }()

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "MyPipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        guard let buffer = device.makeBuffer(bytes: Renderer.triangleVertices,
                                     length: MemoryLayout<AAPLVertex>.stride * Renderer.triangleVertices.count,
                                             options: .storageModeShared) else {
            fatalError("Failed to create vertex buffer.")
        }

        vertices = buffer
        numVertices = Renderer.triangleVertices.count

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = [UInt32(size.width), UInt32(size.height)]
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        commandBuffer.label = "MyCommand"
        renderEncoder.label = "MyRenderEncoder"

        renderEncoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                              width: Double(viewportSize.x),
                                              height: Double(viewportSize.y),
                                              znear: 0, zfar: 0))

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setVertexBuffer(vertices, offset: 0, index: Int(AAPLVertexInputIndexVertices.rawValue))
        renderEncoder.setVertexBytes(&viewportSize,
                                     length: MemoryLayout<simd_uint2>.stride,
                                     index: Int(AAPLVertexInputIndexViewportSize.rawValue))
        renderEncoder.setFragmentTexture(texture, index: Int(AAPLTextureIndexBaseColor.rawValue))
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: numVertices)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
