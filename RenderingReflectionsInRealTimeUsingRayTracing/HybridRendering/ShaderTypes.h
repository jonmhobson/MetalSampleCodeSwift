/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header containing types and enum constants shared between Metal shaders and C/ObjC source
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef enum ConstantIndex {
    ConstantIndexRayTracingEnabled
} ConstantIndex;

typedef enum RTReflectionKernelImageIndex {
    OutImageIndex = 0,
    ThinGBufferPositionIndex = 1,
    ThinGBufferDirectionIndex = 2,
    IrradianceMapIndex = 3
} RTReflectionKernelImageIndex;

typedef enum RTReflectionKernelBufferIndex {
    SceneIndex = 0,
    AccelerationStructureIndex = 1
} RTReflectionKernelBufferIndex;

// The attribute index values that the shader and the C code share to ensure Metal
// shader vertex attribute indices match the Metal API vertex descriptor attribute indices.
typedef enum VertexAttribute {
    VertexAttributePosition     = 0,
    VertexAttributeTexcoord     = 1,
    VertexAttributeNormal       = 2,
    VertexAttributeTangent      = 3,
    VertexAttributeBitangent    = 4
} VertexAttribute;

// The texture index values that the shader and the C code share to ensure
// Metal shader texture indices match indices of Metal API texture set calls
typedef enum TextureIndex {
    TextureIndexBaseColor = 0,
    TextureIndexMetallic = 1,
    TextureIndexRoughness = 2,
    TextureIndexNormal = 3,
    TextureIndexAmbientOcclusion = 4,
    TextureIndexIrradianceMap = 5,
    TextureIndexReflections = 6,
    SkyDomeTexture = 7,
    MaterialTextureCount = TextureIndexAmbientOcclusion + 1 // What?
} TextureIndex;

// The buffer index values that the shader and the C code share to
// ensure Metal shader buffer inputs match Metal API buffer set calls.
typedef enum BufferIndex {
    BufferIndexMeshPositions    = 0,
    BufferIndexMeshGenerics     = 1,
    BufferIndexInstanceTransforms = 2,
    BufferIndexCameraData = 3,
    BufferIndexLightData = 4,
    BufferIndexSubmeshKeypath = 5
} BufferIndex;

typedef struct InstanceTransform {
    matrix_float4x4 modelViewMatrix;
} InstanceTransform;

typedef struct CameraData {
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    vector_float3 cameraPosition;
    float metallicBias;
    float roughnessBias;
} CameraData;

// The structure that the shader and the C code share to ensure the layout o
// data accessed in Metal shaders matches the layout of data set in C code.
typedef struct {
    // Per light properties
    vector_float3 directionalLightInvDirection;
    float lightIntensity;
} LightData;

typedef struct SubmeshKeypath {
    uint32_t instanceID;
    uint32_t submeshID;
} SubmeshKeypath;

#endif /* AAPLShaderTypes_h */
