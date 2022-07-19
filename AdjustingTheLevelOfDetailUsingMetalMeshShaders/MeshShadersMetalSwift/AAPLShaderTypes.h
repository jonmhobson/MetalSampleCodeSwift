/*
See LICENSE folder for this sample’s licensing information.

Abstract:
This file provides definitions that both the app and the shader use.
*/

#pragma once

#ifdef __METAL_VERSION__
#define NSInteger metal::int32_t

using AAPLIndexType = uint16_t;

#else

#define constant
#define constexpr const
#define using typedef
#include <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

typedef enum BufferIndex : int32_t
{
    AAPLBufferIndexMeshVertices = 0,
    AAPLBufferIndexMeshIndices = 1,
    AAPLBufferIndexMeshInfo = 2,
    AAPLBufferIndexFrameData = 3,
    AAPLBufferViewProjectionMatrix = 4,
    AAPLBufferIndexTransforms = 5,
    AAPLBufferIndexMeshColor = 6,
    AAPLBufferIndexLODChoice = 7
} BufferIndex;

typedef struct AAPLVertex
{
    simd_float4 position;
    simd_float4 normal;
    simd_float2 uv;
} AAPLVertex;

typedef struct AAPLIndexRange
{
    // This is the first offset into the indices array.
    uint32_t startIndex;
    // This is one past the first offset into the indices array.
    uint32_t lastIndex;
    // This is the index of the first vertex in the vertex array.
    uint32_t startVertexIndex;
    uint32_t vertexCount;
    uint32_t primitiveCount;
} AAPLIndexRange;

typedef struct AAPLMeshInfo
{
    uint16_t numLODs;
    uint16_t patchIndex;
    simd_float4 color;

    uint16_t vertexCount;

    AAPLIndexRange lod1;
    AAPLIndexRange lod2;
    AAPLIndexRange lod3;
} AAPLMeshInfo;

static constexpr constant uint32_t AAPLNumObjectsX = 16;
static constexpr constant uint32_t AAPLNumObjectsY = 8;
static constexpr constant uint32_t AAPLNumObjectsZ = 1;
static constexpr constant uint32_t AAPLNumObjects = AAPLNumObjectsX * AAPLNumObjectsY * AAPLNumObjectsZ;

static constexpr constant uint32_t AAPLNumPatchSegmentsX = 8;
static constexpr constant uint32_t AAPLNumPatchSegmentsY = 8;

static constexpr constant uint32_t AAPLMaxMeshletVertexCount = 64;
static constexpr constant uint32_t AAPLMaxPrimitiveCount = 126;

static constexpr constant uint32_t AAPLMaxTotalThreadsPerObjectThreadgroup = 1;
static constexpr constant uint32_t AAPLMaxTotalThreadsPerMeshThreadgroup = AAPLMaxPrimitiveCount;
static constexpr constant uint32_t AAPLMaxThreadgroupsPerMeshGrid = 8;

static constexpr constant uint32_t AAPL_FUNCTION_CONSTANT_TOPOLOGY = 0;
