import Foundation

struct Triangle {
    static let vertices: [AAPLVertex] = {
        let triangleSize: Float = 64.0
        return [
            AAPLVertex(position: [-0.5 * triangleSize, -0.5 * triangleSize], color: [1, 1, 1, 1]),
            AAPLVertex(position: [ 0.0 * triangleSize, +0.5 * triangleSize], color: [1, 1, 1, 1]),
            AAPLVertex(position: [+0.5 * triangleSize, -0.5 * triangleSize], color: [1, 1, 1, 1])
        ]
    }()

    var position: simd_float2
    let color: simd_float4
}
