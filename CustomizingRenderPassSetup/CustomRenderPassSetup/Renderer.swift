import Foundation
import MetalKit

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let drawableRenderPipeline: MTLRenderPipelineState

    private let renderTargetTexture: MTLTexture
    private let renderToTextureRenderPipeline: MTLRenderPipelineState
    private let renderToTextureRenderPassDescriptor: MTLRenderPassDescriptor

    var aspectRatio: Float = 1.0

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.25, green: 0.25, blue: 0.25, alpha: 1)

        // Set up a texture for rendering to and sampling from
        let texDescriptor = MTLTextureDescriptor()
        texDescriptor.textureType = .type2D
        texDescriptor.width = 512
        texDescriptor.height = 512
        texDescriptor.pixelFormat = .rgba8Unorm
        texDescriptor.usage = [.renderTarget, .shaderRead]

        guard let rtt = device.makeTexture(descriptor: texDescriptor) else {
            fatalError("Failed to create render target texture")
        }
        self.renderTargetTexture = rtt

        // Set up a render pass descriptor for the render pass to render into
        renderToTextureRenderPassDescriptor = MTLRenderPassDescriptor()
        renderToTextureRenderPassDescriptor.colorAttachments[0].texture = renderTargetTexture
        renderToTextureRenderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderToTextureRenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
        renderToTextureRenderPassDescriptor.colorAttachments[0].storeAction = .store

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "textureVertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "textureFragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Drawable Rendermd Pipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.drawableRenderPipeline = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        guard let rttVertexFunction = defaultLibrary.makeFunction(name: "simpleVertexShader"),
              let rttFragmentFunction = defaultLibrary.makeFunction(name: "simpleFragmentShader") else { fatalError() }

        let rttPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        rttPipelineStateDescriptor.label = "Offscreen Render Pipeline"
        rttPipelineStateDescriptor.vertexFunction = rttVertexFunction
        rttPipelineStateDescriptor.fragmentFunction = rttFragmentFunction
        rttPipelineStateDescriptor.colorAttachments[0].pixelFormat = renderTargetTexture.pixelFormat
        rttPipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.renderToTextureRenderPipeline = {
            do {
                return try device.makeRenderPipelineState(descriptor: rttPipelineStateDescriptor)
            } catch {
                fatalError("Failed to create render to texture pipeline state: \(error)")
            }
        }()

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspectRatio = Float(size.height / size.width)
    }

    static let triangleVertices: [AAPLSimpleVertex] = [
        AAPLSimpleVertex(position: [ 0.5, -0.5], color: [1, 0, 0, 1]),
        AAPLSimpleVertex(position: [-0.5, -0.5], color: [0, 1, 0, 1]),
        AAPLSimpleVertex(position: [ 0.0,  0.5], color: [0, 0, 1, 1])
    ]

    static let quadVertices = [
        AAPLTextureVertex(position: [ 0.5, -0.5], texcoord: [1, 1]),
        AAPLTextureVertex(position: [-0.5, -0.5], texcoord: [0, 1]),
        AAPLTextureVertex(position: [-0.5,  0.5], texcoord: [0, 0]),

        AAPLTextureVertex(position: [ 0.5, -0.5], texcoord: [1, 1]),
        AAPLTextureVertex(position: [-0.5,  0.5], texcoord: [0, 0]),
        AAPLTextureVertex(position: [ 0.5,  0.5], texcoord: [1, 0])
    ]

    private func renderToTexture(renderEncoder: MTLRenderCommandEncoder) {
        renderEncoder.label = "Offscreen Render Pass"
        renderEncoder.setRenderPipelineState(renderToTextureRenderPipeline)
        renderEncoder.setVertexBytes(Renderer.triangleVertices, length: MemoryLayout<AAPLSimpleVertex>.stride * Renderer.triangleVertices.count, index: Int(AAPLVertexInputIndexVertices.rawValue))

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: Renderer.triangleVertices.count)
        renderEncoder.endEncoding()
    }

    private func renderToScreen(renderEncoder: MTLRenderCommandEncoder) {
        renderEncoder.label = "Drawable Render Pass"
        renderEncoder.setRenderPipelineState(drawableRenderPipeline)

        renderEncoder.setVertexBytes(Renderer.quadVertices,
                                     length: Renderer.quadVertices.count * MemoryLayout<AAPLTextureVertex>.stride,
                                     index: Int(AAPLVertexInputIndexVertices.rawValue))

        renderEncoder.setVertexBytes(&aspectRatio,
                                     length: MemoryLayout<Float>.stride,
                                     index: Int(AAPLVertexInputIndexAspectRatio.rawValue))

        renderEncoder.setFragmentTexture(renderTargetTexture, index: Int(AAPLTextureInputIndexColor.rawValue))

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        renderEncoder.endEncoding()
    }

    func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let rttRenderEncoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: renderToTextureRenderPassDescriptor) else { return }

        renderToTexture(renderEncoder: rttRenderEncoder)

        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        renderToScreen(renderEncoder: renderEncoder)

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
