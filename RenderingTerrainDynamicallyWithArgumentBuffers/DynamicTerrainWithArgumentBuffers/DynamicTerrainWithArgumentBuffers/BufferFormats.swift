import Metal

enum BufferFormats {
    static let gBuffer0Format = MTLPixelFormat.bgra8Unorm_srgb
    static let gBuffer1Format = MTLPixelFormat.rgba8Unorm

    static let depthFormat = MTLPixelFormat.depth32Float
    static let shadowDepthFormat = MTLPixelFormat.depth32Float

    static let backBufferFormat = MTLPixelFormat.bgra8Unorm_srgb
}
