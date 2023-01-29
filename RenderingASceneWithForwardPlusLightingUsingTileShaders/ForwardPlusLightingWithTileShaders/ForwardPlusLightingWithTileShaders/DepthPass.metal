#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "ShaderCommon.h"

struct VertexOutput {
    float4 position [[position]];
};

// A depth pre-pass is necessary in forward plus rendering to produce
// minum and maximum depth bounds for light culling.
vertex VertexOutput depth_pre_pass_vertex(Vertex in [[ stage_in ]],
                                          constant FrameData & frameData [[ buffer(BufferIndexFrameData) ]]) {
    // Make the position a float4 to perform 4x4 matrix math on it.
    VertexOutput out;
    float4 position = float4(in.position, 1.0);

    // Calculate the position in clip space.
    out.position = frameData.projectionMatrix * frameData.modelViewMatrix * position;

    return out;
}

fragment ColorData depth_pre_pass_fragment(VertexOutput in [[stage_in]]) {
    // Populate on-tile geometry buffer data.
    ColorData f;

    // Setting color in the depth pre-pass is unnecessary, but may make debugging easier.
    // f.lighting = half4(0, 0, 0, 1);

    // Set the depth in clip space, which you use in Culling to perform per-tile light culling
    f.depth = in.position.z;
    return f;
}
