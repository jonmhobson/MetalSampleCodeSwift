import Foundation
import MetalKit

@MainActor
final class Renderer: NSObject {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue

    init(device: any MTLDevice) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue
    }

    func draw(in view: MTKView) {
        // The render pass descriptor references the texture into which Metal should draw
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        commandEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
