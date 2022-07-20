import SwiftUI
import MetalKit

class MetalViewInteractor: ObservableObject {
    let renderer: Renderer
    let metalView = MTKView()

    init() {
        renderer = Renderer(metalView: metalView)
    }

    @Published var leftZOffset: Float = 0.25 {
        didSet { renderer.leftVertexDepth = leftZOffset }
    }

    @Published var topZOffset: Float = 0.25 {
        didSet { renderer.topVertexDepth = topZOffset }
    }

    @Published var rightZOffset: Float = 0.25 {
        didSet { renderer.rightVertexDepth = rightZOffset }
    }
}

struct MetalView: View {
    @StateObject var viewInteractor = MetalViewInteractor()

    var body: some View {
        ZStack {
            MetalViewRepresentable(metalView: viewInteractor.metalView)
            VStack {
                Slider(value: $viewInteractor.topZOffset).frame(width: 100)
                Text("\(viewInteractor.topZOffset, specifier: "%.2f")").padding(.top, 24)
                Spacer()
                HStack {
                    Slider(value: $viewInteractor.leftZOffset).frame(width: 100)
                    Text("\(viewInteractor.leftZOffset, specifier: "%.2f")").offset(x: -50, y: -50)
                    Spacer()
                    Text("\(viewInteractor.rightZOffset, specifier: "%.2f")").offset(x: 50, y: -50)
                    Slider(value: $viewInteractor.rightZOffset).frame(width: 100)
                }
            }
            .padding()
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
