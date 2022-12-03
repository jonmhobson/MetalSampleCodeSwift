#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Buffer index values shared between shader and C code to ensure Metal shader buffer inputs match
// Metal API buffer set calls
typedef enum VertexBufferIndex {
    VertexBufferIndexVertices = 0
} VertexBufferIndex;

// Buffer index values shared between shader and C code to ensure Metal shader buffer inputs match
// Metal API buffer set calls
typedef enum FragmentBufferIndex {
    FragmentBufferIndexArguments = 0
} FragmentBufferIndex;

// Constant values shared between shader and C code which idicate the size of argument arrays
// in the structure defining the argument buffers
typedef enum NumArguments {
    NumBufferArguments = 30,
    NumTextureArguments = 32
} NumArguments;

// Defines the layout of each vertex in the array of vertices set as an input to our
// Metal vertex shader
typedef struct Vertex {
    vector_float2 position;
    vector_float2 texCoord;
} Vertex;

#ifndef __METAL_VERSION__

#include <Metal/Metal.h>

typedef struct FragmentShaderArguments {
    MTLResourceID   exampleTextures[NumTextureArguments];
    uint64_t        exampleBuffers[NumBufferArguments];
    uint32_t        exampleConstants[NumBufferArguments];
} FragmentShaderArguments;

#else
struct FragmentShaderArguments {
    array<texture2d<float>, NumTextureArguments>    exampleTextures;
    array<device float *, NumBufferArguments>       exampleBuffers;
    array<uint32_t, NumBufferArguments>             exampleConstants;
};
#endif

#endif /* ShaderTypes_h */
