import SwiftUI
import MetalKit

class MetalViewInteractor: ObservableObject {
    let renderer: Renderer
    let scene: Scene
    let metalView: MTKView

    init() {
        metalView = MTKView()

        let devices = MTLCopyAllDevices()

        guard let device = devices.first(where: { $0.supportsRaytracing }) else {
            fatalError("No device supporting raytracing found")
        }

        metalView.device = device
        metalView.colorPixelFormat = .rgba16Float

        scene = Scene(device: metalView.device!)
        renderer = Renderer(device: device, scene: scene)
        metalView.delegate = renderer
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let metalView: MTKView

    func makeNSView(context: Context) -> some NSView {
        metalView
    }

    func updateNSView(_ nsView: NSViewType, context: Context) {
        updateMetalView()
    }

    func updateMetalView() {}
}

struct MetalView: View {
    @StateObject var viewInteractor = MetalViewInteractor()

    var body: some View {
        ZStack {
            MetalViewRepresentable(metalView: viewInteractor.metalView)
        }
    }
}

struct MetalView_Previews: PreviewProvider {
    static var previews: some View {
        MetalView()
    }
}
