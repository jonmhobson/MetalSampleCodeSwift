import Foundation
import simd
import Metal

extension simd_float4x4 {
    init(_ m00: Float, _ m10: Float, _ m20: Float, _ m30: Float,
         _ m01: Float, _ m11: Float, _ m21: Float, _ m31: Float,
         _ m02: Float, _ m12: Float, _ m22: Float, _ m32: Float,
         _ m03: Float, _ m13: Float, _ m23: Float, _ m33: Float) {
        self.init([m00, m01, m02, m03],
                  [m10, m11, m12, m13],
                  [m20, m21, m22, m23],
                  [m30, m31, m32, m33])
    }

    static func perspectiveRightHand(fovyRadians: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let ys = 1.0 / tanf(fovyRadians * 0.5)
        let xs = ys / aspect
        let zs = farZ / (nearZ - farZ)
        return simd_float4x4(xs, 0, 0, 0,
                             0, ys, 0, 0,
                             0, 0, zs, nearZ * zs,
                             0, 0, -1, 0)
    }

    static func rotation(_ radians: Float, _ axis: SIMD3<Float>) -> simd_float4x4 {
        let axis = simd_normalize(axis)
        let ct = cosf(radians)
        let st = sinf(radians)
        let ci = 1 - ct
        let x = axis.x
        let y = axis.y
        let z = axis.z

        return simd_float4x4(ct + x * x * ci, x * y * ci - z * st, x * z * ci + y * st, 0,
                             y * x * ci + z * st, ct + y * y * ci, y * z * ci - x * st, 0,
                             z * x * ci - y * st, z * y * ci + x * st, ct + z * z * ci, 0,
                             0, 0, 0, 1)
    }

    static func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
        return simd_float4x4(1, 0, 0, t.x,
                             0, 1, 0, t.y,
                             0, 0, 1, t.z,
                             0, 0, 0, 1)
    }

    func dropLastRow() -> MTLPackedFloat4x3 {
        return MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(columns.0.x, columns.0.y, columns.0.z),
                                           MTLPackedFloat3Make(columns.1.x, columns.1.y, columns.1.z),
                                           MTLPackedFloat3Make(columns.2.x, columns.2.y, columns.2.z),
                                           MTLPackedFloat3Make(columns.3.x, columns.3.y, columns.3.z)))
    }
}
