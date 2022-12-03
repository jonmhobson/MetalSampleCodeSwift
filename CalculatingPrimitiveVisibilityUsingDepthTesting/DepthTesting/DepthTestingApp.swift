import SwiftUI

@main
struct DepthTestingApp: App {
    var body: some Scene {
        Window("Depth Testing", id: "depth-testing") {
            MetalView()
        }
    }
}
