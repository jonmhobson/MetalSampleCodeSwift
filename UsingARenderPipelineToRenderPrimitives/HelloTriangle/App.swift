import SwiftUI

@main
struct HelloTriangleApp: App {
    var body: some Scene {
        Window("Hello Triangle", id: "hello-triangle"){
            MetalView()
        }
    }
}
