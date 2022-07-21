import SwiftUI
import MetalKit

class MetalViewInteractor: ObservableObject {
    let renderer: Renderer
    let metalView = MTKView()

    init() {
        renderer = Renderer(metalView: metalView)
    }
}

struct MetalView: View {
    @StateObject var viewInteractor = MetalViewInteractor()

    var body: some View {
        VStack {
            MetalViewRepresentable(metalView: viewInteractor.metalView)
        }
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let metalView: MTKView
    func makeNSView(context: Context) -> some NSView { metalView }
    func updateNSView(_ nsView: NSViewType, context: Context) { updateMetalView() }
    func updateMetalView() {}
}

struct MetalView_Previews: PreviewProvider {
    static var previews: some View {
        MetalView()
    }
}
