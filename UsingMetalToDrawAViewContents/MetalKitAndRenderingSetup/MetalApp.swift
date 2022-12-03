import SwiftUI

@main
struct MetalApp: App {
    var body: some Scene {
        Window("Metal Rendering", id: "metal-rendering") {
            MetalView()
        }
    }
}
