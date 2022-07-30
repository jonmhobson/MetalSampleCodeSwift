#ifndef AAPLShaderTypes_h
#define AAPLShaderTypes_h

#include <simd/simd.h>

// The buffer index values that the shader and C code share to ensure Metal
//   vertex shader buffer inputs match Metal API set calls.
typedef enum AAPLVertexBufferIndex
{
    AAPLVertexBufferIndexVertices = 0,
} AAPLVertexBufferIndex;

// The buffer index values that the shader and C code share to ensure Metal
//   fragment shader buffer inputs match Metal API set calls.
typedef enum AAPLFragmentBufferIndex
{
    AAPLFragmentBufferIndexArguments = 0,
} AAPLFragmentBufferIndex;

//  Defines the layout of each vertex in the array of vertices that functions
//     as an input to the Metal vertex shader.
typedef struct AAPLVertex {
    vector_float2 position;
    vector_float2 texCoord;
    vector_float4 color;
} AAPLVertex;

#ifndef __METAL_VERSION__

typedef uint16_t half;

#else

struct FragmentShaderArguments {
    texture2d<half>  exampleTexture;
    sampler          exampleSampler;
    device float    *exampleBuffer;
    uint32_t         exampleConstant;
};

#endif

#endif /* AAPLShaderTypes_h */
