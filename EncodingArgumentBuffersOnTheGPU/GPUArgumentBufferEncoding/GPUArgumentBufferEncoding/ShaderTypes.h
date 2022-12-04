#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#define QuadSize        0.3
#define QuadSpacing     0.31
#define GridWidth       11
#define NumInstances    66
#define NumTextures     67
#define GridHeight      ((NumInstances+1) / GridWidth)

// The buffer index values that the shader and C code share to ensure Metal
//   vertex shader buffer inputs match Metal API set calls.
typedef enum VertexBufferIndex
{
    VertexBufferIndexVertices,
    VertexBufferIndexInstanceParams,
    VertexBufferIndexFrameState
} VertexBufferIndex;

// The buffer index values that the shader and C code share to ensure Metal
//   fragment shader buffer inputs match Metal API set calls.
typedef enum FragmentBufferIndex {
    FragmentBufferIndexInstanceParams,
    FragmentBufferIndexFrameState
} FragmentBufferIndex;

typedef enum ComputeBufferIndex {
    ComputeBufferIndexSourceTextures,
    ComputeBufferIndexInstanceParams,
    ComputeBufferIndexFrameState
} ComputeBufferIndex;

typedef enum ArgumentBufferID {
    ArgumentBufferIDTexture = 0
} ArgumentBufferID;

//  Defines the layout of each vertex in the array of vertices that functions
//     as an input to the Metal vertex shader.
typedef struct Vertex {
    vector_float2 position;
    vector_float2 texCoord;
} Vertex;

typedef struct FrameState {
    uint            textureIndexOffset;
    float           slideFactor;
    vector_float2   offset;
    vector_float2   quadScale;
} FrameState;

#endif /* ShaderTypes_h */
