import SwiftUI

@main
struct BasicTexturingApp: App {
    var body: some Scene {
        Window("Basic Texturing", id: "basic-texturing") {
            MetalView()
        }
    }
}
