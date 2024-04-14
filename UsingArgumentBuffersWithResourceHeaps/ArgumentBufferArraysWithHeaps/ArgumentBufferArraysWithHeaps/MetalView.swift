import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        private let renderer: Renderer
        let device: MTLDevice

        init(_ parent: MetalView) {
            guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
            self.device = device
            self.renderer = Renderer(device: device)
        }

        // Swift 6 concurrency note: We have to do a little nonisolated / MainActor.assumeIsolated
        // dance here because these delegate functions will always be called on the main thread.
        // but the protocol does not annotate this actor conformance.
        nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            MainActor.assumeIsolated {
                renderer.mtkView(view, drawableSizeWillChange: size)
            }
        }

        nonisolated func draw(in view: MTKView) {
            MainActor.assumeIsolated {
                renderer.draw(in: view)
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> some NSView {
        let metalView = MTKView()

        metalView.device = context.coordinator.device
        metalView.delegate = context.coordinator
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.5, blue: 1, alpha: 1)

        return metalView
    }

    func updateNSView(_ nsView: NSViewType, context: Context) {}
}

#Preview {
    MetalView()
}
