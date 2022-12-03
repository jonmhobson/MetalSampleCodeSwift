import Foundation
import MetalKit

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState

    private let depthState: MTLDepthStencilState

    private var viewportSize: simd_uint2 = [0, 0]

    var leftVertexDepth: Float = 0.25
    var topVertexDepth: Float = 0.25
    var rightVertexDepth: Float = 0.25

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0.25, green: 0.25, blue: 0.25, alpha: 1)

        metalView.depthStencilPixelFormat = .depth32Float
        metalView.clearDepth = 1.0

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Render Pipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat
        pipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .lessEqual
        depthDescriptor.isDepthWriteEnabled = true

        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            fatalError("Failed to create depth stencil state")
        }

        self.depthState = depthState

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

        commandBuffer.label = "Command Buffer"
        renderEncoder.label = "Render Encoder"

        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(depthState)

        renderEncoder.setVertexBytes(&viewportSize,
                                     length: MemoryLayout<simd_uint2>.stride,
                                     index: Int(AAPLVertexInputIndexViewport.rawValue))

        let vx = Float(viewportSize.x)
        let vy = Float(viewportSize.y)

        let quadVertices: [AAPLVertex] = [
            AAPLVertex(position: [     100,      100, 0.5], color: [0.5, 0.5, 0.5, 1]),
            AAPLVertex(position: [     100, vy - 100, 0.5], color: [0.5, 0.5, 0.5, 1]),
            AAPLVertex(position: [vx - 100, vy - 100, 0.5], color: [0.5, 0.5, 0.5, 1]),

            AAPLVertex(position: [     100,      100, 0.5], color: [0.5, 0.5, 0.5, 1]),
            AAPLVertex(position: [vx - 100, vy - 100, 0.5], color: [0.5, 0.5, 0.5, 1]),
            AAPLVertex(position: [vx - 100,      100, 0.5], color: [0.5, 0.5, 0.5, 1])
        ]

        renderEncoder.setVertexBytes(quadVertices,
                                     length: MemoryLayout<AAPLVertex>.stride * quadVertices.count,
                                     index: 0)

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        let triangleVertices: [AAPLVertex] = [
            AAPLVertex(position: [     200, vy - 200,  leftVertexDepth], color: [1, 1, 1, 1]),
            AAPLVertex(position: [vx * 0.5,      200,   topVertexDepth], color: [1, 1, 1, 1]),
            AAPLVertex(position: [vx - 200, vy - 200, rightVertexDepth], color: [1, 1, 1, 1]),
        ]

        renderEncoder.setVertexBytes(triangleVertices, length: MemoryLayout<AAPLVertex>.stride * triangleVertices.count, index: 0)

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
