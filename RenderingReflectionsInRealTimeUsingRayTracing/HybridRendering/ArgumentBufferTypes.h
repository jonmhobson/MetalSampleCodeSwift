#ifndef ArgumentBufferTypes_h
#define ArgumentBufferTypes_h

#include "ShaderTypes.h"

typedef enum ArgumentBufferID {
    ArgumentBufferIDGenericsTexcoord,
    ArgumentBufferIDGenericsNormal,
    ArgumentBufferIDGenericsTangent,
    ArgumentBufferIDGenericsBitangent,

    ArgumentBufferIDSubmeshIndices,
    ArgumentBufferIDSubmeshMaterials,

    ArgumentBufferIDMeshPositions,
    ArgumentBufferIDMeshGenerics,
    ArgumentBufferIDMeshSubmeshes,

    ArgumentBufferIDInstanceMesh,
    ArgumentBufferIDInstanceTransform,

    ArgumentBufferIDSceneInstances,
    ArgumentBufferIDSceneMeshes,
} ArgumentBufferID;

#if __METAL_VERSION__

#include <metal_stdlib>
using namespace metal;

struct MeshGenerics {
    float2  texcoord    [[ id(ArgumentBufferIDGenericsTexcoord) ]];
    half4   normal      [[ id(ArgumentBufferIDGenericsNormal) ]];
    half4   tangent     [[ id(ArgumentBufferIDGenericsTangent) ]];
    half4   bitangent   [[ id(ArgumentBufferIDGenericsBitangent) ]];
};

struct Submesh {
    // The container mesh stores positions and generic vertex attribute arrays.
    // The submesh stores only indices into these vertex arrays.
    uint32_t shortIndexType [[id(0)]];

    // The indices for the container mesh's position and generics arrays.
    constant uint32_t*                              indices [[ id(ArgumentBufferIDSubmeshIndices) ]];

    // The fixed size array of material textures.
    array<texture2d<float>, MaterialTextureCount>   materials [[ id(ArgumentBufferIDSubmeshMaterials) ]];
};

struct Mesh {
    // The arrays of vertices.
    constant packed_float3* positions   [[ id(ArgumentBufferIDMeshPositions)]];
    constant MeshGenerics* generics     [[ id(ArgumentBufferIDMeshGenerics) ]];

    // The array of submeshes.
    constant Submesh* submeshes         [[ id(ArgumentBufferIDMeshSubmeshes)]];
};

struct Instance {
    // A reference to a single mesh in the meshes array stored in structure 'Scene'.
    uint32_t meshIndex [[id(0)]];

    float4x4 transform [[id(1)]];
};

struct Scene {
    // The array of instances.
    constant Instance* instances [[ id(ArgumentBufferIDSceneInstances) ]];
    constant Mesh* meshes [[ id(ArgumentBufferIDSceneMeshes) ]];
};

#else

#include <Metal/Metal.h>

struct Submesh {
    uint32_t shortIndexType;
    uint64_t indices;
    MTLResourceID materials[MaterialTextureCount];
};

struct Mesh {
    uint64_t positions;
    uint64_t generics;

    uint64_t submeshes;
};

struct Instance {
    uint32_t meshIndex;
    matrix_float4x4 transform;
};

struct Scene3D {
    uint64_t instances;
    uint64_t meshes;
};

#endif

#endif /* ArgumentBufferTypes_h */
