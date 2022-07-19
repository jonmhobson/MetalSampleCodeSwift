import SwiftUI
import MetalKit

enum Speed: Float {
    case off = 0.0
    case slow = 0.1
    case fast = 0.25
}

enum Topology: Int {
    case points = 0
    case lines = 1
    case triangles = 2
}

enum DetailLevel: Int {
    case low = 2
    case medium = 1
    case high = 0
}

class MetalViewInteractor: ObservableObject {
    var renderer: Renderer? = nil

    @Published var topology: Topology = .triangles {
        didSet {
            renderer?.topologyChoice = topology.rawValue
        }
    }

    @Published var detail: DetailLevel = .high {
        didSet {
            renderer?.lodChoice = detail.rawValue
        }
    }

    @Published var speed: Speed = .slow {
        didSet {
            renderer?.rotationSpeed = speed.rawValue
        }
    }

    @Published var translation: Float = 0.0 {
        didSet {
            renderer?.offsetZ = translation
        }
    }

    let metalView = MTKView()

    init() {
        renderer = Renderer(metalView: metalView)
        renderer?.offsetY = -1.5
    }
}

struct MetalView: View {

    @StateObject var viewInteractor = MetalViewInteractor()

    var body: some View {
        ZStack(alignment: .top) {
            MetalViewRepresentable(metalView: viewInteractor.metalView)

            HStack {
                VStack {
                    Picker("Detail", selection: $viewInteractor.detail) {
                        Text("Low").tag(DetailLevel.low)
                        Text("Medium").tag(DetailLevel.medium)
                        Text("High").tag(DetailLevel.high)
                    }
                    .pickerStyle(.segmented)

                    Picker("Topology", selection: $viewInteractor.topology) {
                        Text("Points").tag(Topology.points)
                        Text("Lines").tag(Topology.lines)
                        Text("Triangles").tag(Topology.triangles)
                    }
                    .pickerStyle(.segmented)
                }
                .frame(width: 300)
                .padding()

                Spacer()

                VStack {
                    Picker("Rotation", selection: $viewInteractor.speed) {
                        Text("Off").tag(Speed.off)
                        Text("Slow").tag(Speed.slow)
                        Text("Normal").tag(Speed.fast)
                    }
                    .pickerStyle(.segmented)
                    Slider(value: $viewInteractor.translation) { Text("Translation") }
                }
                .frame(width: 300)
                .padding()
            }
        }
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

struct MetalView_Previews: PreviewProvider {
    static var previews: some View {
        VStack {
            MetalView()
        }
    }
}
