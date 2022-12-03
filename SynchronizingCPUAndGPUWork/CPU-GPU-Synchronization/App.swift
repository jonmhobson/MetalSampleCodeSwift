import SwiftUI

@main
struct SyncApp: App {
    var body: some Scene {
        Window("CPU-GPU Sync", id: "cpu-gpu-sync") {
            MetalView()
        }
    }
}
