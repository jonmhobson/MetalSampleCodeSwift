import SwiftUI
import MetalKit

class MyMTKView: MTKView {
    override var acceptsFirstResponder: Bool { true }
}

class MetalViewInteractor: ObservableObject {
    let renderer: MainRenderer
    let metalView = MyMTKView()

    init() {
        renderer = MainRenderer(metalView: metalView)
    }
}

struct MetalView: View {
    @StateObject var viewInteractor = MetalViewInteractor()

    var body: some View {
        VStack {
            MetalViewRepresentable(metalView: viewInteractor.metalView, renderer: viewInteractor.renderer)
        }
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let metalView: MTKView
    let renderer: MainRenderer

    func makeCoordinator() -> Coordinator {
        let coordinator = Coordinator(renderer: renderer, size: metalView.drawableSize)
        return coordinator
    }

    class Coordinator: NSResponder {
        let renderer: MainRenderer
        let size: CGSize

        init(renderer: MainRenderer, size: CGSize) {
            self.renderer = renderer
            self.size = size
            super.init()
        }

        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }

        override func mouseExited(with event: NSEvent) {
            renderer.cursorPosition = [-1, -1]
        }

        override func rightMouseDown(with event: NSEvent) {
            renderer.mouseButtonMask |= 2
        }

        override func rightMouseUp(with event: NSEvent) {
            renderer.mouseButtonMask &= (~2)
        }

        override func mouseDown(with event: NSEvent) {
            renderer.mouseButtonMask |= 1
        }

        override func mouseUp(with event: NSEvent) {
            renderer.mouseButtonMask &= (~1)
        }

        override func mouseMoved(with event: NSEvent) {
            renderer.cursorPosition = [Float(event.locationInWindow.x),
                                       Float(size.height - event.locationInWindow.y)]
        }

        override func mouseDragged(with event: NSEvent) {
            renderer.mouseDrag = [Float(event.deltaX), Float(event.deltaY)]
        }

        override func rightMouseDragged(with event: NSEvent) {
            renderer.mouseDrag = [Float(event.deltaX), Float(event.deltaY)]
        }

        override func keyUp(with event: NSEvent) {
            renderer.pressedKeys.remove(Controls(rawValue: UInt8(event.keyCode))!)
        }

        override func keyDown(with event: NSEvent) {
            if !event.isARepeat {
                renderer.pressedKeys.insert(Controls(rawValue: UInt8(event.keyCode))!)
            }
        }
    }

    func makeNSView(context: Context) -> NSView {
        let options: NSTrackingArea.Options = [
            .activeAlways,
            .inVisibleRect,
            .mouseEnteredAndExited,
            .mouseMoved
        ]

        let trackingArea = NSTrackingArea(rect:metalView.frame,
                                          options: options,
                                          owner: context.coordinator)

        metalView.addTrackingArea(trackingArea)

        DispatchQueue.main.async {
            metalView.window?.makeFirstResponder(metalView)
        }

        return metalView
    }
    func updateNSView(_ nsView: NSView, context: Context) { updateMetalView() }
    func updateMetalView() {}

    static func dismantleNSView(_ nsView: NSView, coordinator: ()) {
        nsView.trackingAreas.forEach { nsView.removeTrackingArea($0) }
    }
}

struct MetalView_Previews: PreviewProvider {
    static var previews: some View {
        MetalView()
    }
}
