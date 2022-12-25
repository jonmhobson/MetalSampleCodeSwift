import Foundation
import MetalKit

let kMaxBuffersInFlight = 3

enum Controls: UInt8 {
    // Translation
    case forward = 0x0d // W key
    case backward = 0x01 // S key
    case strafeUp = 0x31 // Spacebar
    case strafeDown = 0x08 // C key
    case strafeLeft = 0x00 // A key
    case strafeRight = 0x02 // D key

    // Rotation
    case rollLeft = 0x0c // Q key
    case rollRight = 0x0e // E key
    case turnLeft = 0x7b // Left arrow
    case turnRight = 0x7c // Right arrow
    case turnUp = 0x7e // Up arrow
    case turnDown = 0x7d // Down arrow

    // Brush size
    case incBrush = 0x1e // Right bracket
    case decBrush = 0x21 // Left bracket

    // Additional virtual keys
    case fast = 0x80
    case slow = 0x81
}

final class MainRenderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.framebufferOnly = false
        metalView.colorPixelFormat = BufferFormats.backBufferFormat

        super.init()

        metalView.delegate = self
    }

    // Cursor position in pixel coordinates
    var cursorPosition: simd_float2 = [-1, -1]
    var mouseButtonMask: UInt = 0
    var brushSize: Float = 1.0
    var pressedKeys = Set<Controls>()
    var mouseDrag: simd_float2 = .zero

    let inFlightSemaphore = DispatchSemaphore(value: kMaxBuffersInFlight)
}

extension MainRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

    func draw(in view: MTKView) {
        inFlightSemaphore.wait()

        if let drawable = view.currentDrawable,
           let renderPassDescriptor = view.currentRenderPassDescriptor,
           let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.label = "Frame CB"

            commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
                inFlightSemaphore.signal()
            }

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}
