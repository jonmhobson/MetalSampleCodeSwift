import SwiftUI

@main
struct GPUArgumentBufferEncodingApp: App {
    var body: some Scene {
        Window("GPU Argument Buffer Encoding", id: "gpu-argument-buffer-encoding"){
            MetalView()
        }
    }
}
