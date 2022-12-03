import SwiftUI

@main
struct TextureComputeApp: App {
    var body: some Scene {
        Window("Texture Compute", id: "texture-compute") {
            MetalView()
        }
    }
}
