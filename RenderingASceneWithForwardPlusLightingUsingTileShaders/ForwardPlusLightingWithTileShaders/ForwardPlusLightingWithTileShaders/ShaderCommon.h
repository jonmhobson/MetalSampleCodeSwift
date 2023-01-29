#ifndef ShaderCommon_h
#define ShaderCommon_h

// Per-tile data computed by the culling kernel.
struct TileData {
    atomic_int numLights;
    float minDepth;
    float maxDepth;
};

// Per-vertex inputs populated by the vertex buffer laid out with the MTLVertexDescriptor Metal API
struct Vertex {
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
    half3 normal [[attribute(VertexAttributeNormal)]];
    half3 tangent [[attribute(VertexAttributeTangent)]];
    half3 bitangent [[attribute(VertexAttributeBitangent)]];
};

// Outputs for the color attachments.
struct ColorData {
    half4 lighting [[color(RenderTargetLighting)]];
    float depth [[color(RenderTargetDepth)]];
};

#endif /* ShaderCommon_h */
