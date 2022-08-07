#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"

using namespace metal;

typedef struct {
    float4 position [[position]];
    float2 texCoord;
} RasterizerData;

// Vertex shader function
vertex RasterizerData texturedQuadVertex(uint vertexID[[vertex_id]],
                                         const device AAPLVertex *vertices [[buffer(AAPLVertexBufferIndexVertices)]],
                                         constant float2& quadScale [[buffer(AAPLVertexBufferIndexScale)]]) {
    RasterizerData out;

    float2 position = vertices[vertexID].position * quadScale;

    out.position.xy = position;
    out.position.z = 0.0;
    out.position.w = 1.0;

    out.texCoord = vertices[vertexID].texCoord;

    return out;
}

// Fragment shader function
fragment half4 texturedQuadFragment(RasterizerData in [[stage_in]],
                                    texture2d<half> texture [[texture(AAPLFragmentTextureIndexImage)]],
                                    constant float &mipmapBias [[buffer(AAPLFragmentBufferIndexMipBias)]]) {
    constexpr sampler sampler(min_filter::linear,
                              mag_filter::linear,
                              mip_filter::linear);

    half4 color = texture.sample(sampler, in.texCoord, level(mipmapBias));
    return color;
}
