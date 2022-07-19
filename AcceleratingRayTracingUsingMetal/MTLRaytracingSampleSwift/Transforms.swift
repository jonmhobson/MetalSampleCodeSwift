import Foundation
import simd

func matrix4x4_translation(tx: Float, ty: Float, tz: Float) -> matrix_float4x4 {
    return simd_matrix_from_rows(simd_make_float4(1, 0, 0, tx),
                                 simd_make_float4(0, 1, 0, ty),
                                 simd_make_float4(0, 0, 1, tz),
                                 simd_make_float4(0, 0, 0, 1))
}

func matrix4x4_rotation(radians: Float, axis: vector_float3) -> matrix_float4x4 {
    let axis = simd_normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1.0 - ct
    let x = axis.x,
        y = axis.y,
        z = axis.z

    return simd_matrix_from_rows(simd_make_float4(ct + x * x * ci, x * y * ci - z * st, x * z * ci + y * st, 0),
                                 simd_make_float4(y * x * ci + z * st, ct + y * y * ci, y * z * ci - x * st, 0),
                                 simd_make_float4(z * x * ci - y * st, z * y * ci + x * st, ct + z * z * ci, 0),
                                 simd_make_float4(0, 0, 0, 1))
}

func matrix4x4_scale(sx: Float, sy: Float, sz: Float) -> matrix_float4x4 {
    return simd_matrix_from_rows(simd_make_float4(sx,  0,  0, 0),
                                 simd_make_float4( 0, sy,  0, 0),
                                 simd_make_float4( 0,  0, sz, 0),
                                 simd_make_float4( 0,  0,  0, 1))
}
