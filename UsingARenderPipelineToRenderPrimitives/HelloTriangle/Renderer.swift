import Foundation
import MetalKit

final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState

    var viewportSize: simd_uint2 = [0, 0]

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Simple Pipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat

        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = [UInt32(size.width), UInt32(size.height)]
    }

    static let triangleVertices: [AAPLVertex] = [
        AAPLVertex(position: [ 250, -250], color: [1, 0, 0, 1]),
        AAPLVertex(position: [-250, -250], color: [0, 1, 0, 1]),
        AAPLVertex(position: [   0,  250], color: [0, 0, 1, 1])
    ]

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

        renderEncoder.setVertexBytes(Renderer.triangleVertices,
                                     length: Renderer.triangleVertices.count * MemoryLayout<AAPLVertex>.stride,
                                     index: Int(AAPLVertexInputIndexVertices.rawValue))

        renderEncoder.setVertexBytes(&viewportSize,
                                     length: MemoryLayout<simd_uint2>.stride,
                                     index: Int(AAPLVertexInputIndexViewportSize.rawValue))

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
