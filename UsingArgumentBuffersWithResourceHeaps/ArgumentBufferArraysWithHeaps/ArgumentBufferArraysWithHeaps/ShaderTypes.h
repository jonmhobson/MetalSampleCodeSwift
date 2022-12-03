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

// Argument buffer indices shared between shader and C code to ensure Metal shader buffer
// input matches Metal API texture set calls
typedef enum ArgumentBufferID {
    ArgumentBufferIDExampleTextures     = 0,
    ArgumentBufferIDExampleBuffers      = 100,
    ArgumentBufferIDExampleConstants    = 200
} ArgumentBufferID;

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

#endif /* ShaderTypes_h */
