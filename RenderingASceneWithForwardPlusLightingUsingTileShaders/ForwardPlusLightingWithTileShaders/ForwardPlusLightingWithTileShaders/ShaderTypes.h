#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Buffer index values shared between shader and Swift code to ensure that Metal shader buffer inputs
// match Metal API buffer set calls.
typedef enum BufferIndices {
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics = 1,
    BufferIndexFrameData = 2,
    BufferIndexLightsData = 3,
    BufferIndexLightsPosition = 4
} BufferIndices;

// Attribute index values shared between shader and Swift code to ensure that Metal shader vertex
// attribute indices match Metal API ertex descriptor attribute indices.
typedef enum VertexAttributes {
    VertexAttributePosition = 0,
    VertexAttributeTexcoord = 1,
    VertexAttributeNormal = 2,
    VertexAttributeTangent = 3,
    VertexAttributeBitangent = 4
} VertexAttributes;

// Texture index values shared between shader and Swift code to ensure that Metal shader texture
// indices match Metal API texture set calls.
typedef enum TextureIndices {
    TextureIndexBaseColor = 0,
    TextureIndexSpecular = 1,
    TextureIndexNormal = 2,

    NumTextureIndices
} TextureIndices;

// Threadgroup space buffer indices.
typedef enum ThreadgroupIndices {
    ThreadgroupBufferIndexLightList = 0,
    ThreadgroupBufferIndexTileData = 1
} ThreadgroupIndices;

typedef enum RenderTargetIndices {
    RenderTargetLighting = 0, // Required for procedural blending.
    RenderTargetDepth = 1
} RenderTargetIndices;

// Structures shared between shader and Swift code to ensure the layout of uniform data accessed in
// Metal shaders matches the layout of frame data set in C code.

// Per-light characteristics.
typedef struct {
    vector_float3 lightColor;
    float lightRadius;
    float lightSpeed;
} PointLight;

// Data constant across all threads, vertices, and fragments.
typedef struct {
    // Per-frame constants.
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 projectionMatrixInv;
    matrix_float4x4 viewMatrix;
    matrix_float4x4 viewMatrixInv;
    vector_float2 depthUnproject;
    vector_float3 screenToViewSpace;

    // Per-mesh constants
    matrix_float4x4 modelViewMatrix;
    matrix_float3x3 normalMatrix;
    matrix_float4x4 modelMatrix;

    // Per-light properties.
    vector_float3 ambientLightColor;
    vector_float3 directionalLightDirection;
    vector_float3 directionalLightColor;
    uint framebufferWidth;
    uint framebufferHeight;
} FrameData;

// Simple vertex used to render the fairies.
typedef struct {
    vector_float2 position;
} SimpleVertex;

#define NumSamples 4
#define NumLights 1024
#define MaxLightsPerTile 64
#define TileWidth 16
#define TileHeight 16

// Size of an on-tile structure containing information such as maximum tile depth, minimum tile
// depth, and a list of lights in the tile.
#define TileDataSize 256

// Temporary buffer used for depth reduction.
// Buffer size needs to be at least tile width * tile height * 4
#define ThreadgroupBufferSize MAX(MaxLightsPerTile * sizeof(uint32_t), TileWidth * TileHeight * sizeof(uint32_t))

#endif /* AAPLShaderTypes_h */
