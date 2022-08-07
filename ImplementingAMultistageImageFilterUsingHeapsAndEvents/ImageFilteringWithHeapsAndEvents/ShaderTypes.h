#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef enum AAPLBlurBufferIndex {
    AAPLBlurBufferIndexLOD = 0,
} AAPLBlurBufferIndex;

typedef enum AAPLVertexBufferIndex {
    AAPLVertexBufferIndexVertices = 0,
    AAPLVertexBufferIndexScale = 1
} AAPLVertexBufferIndex;

typedef enum AAPLFragmentBufferIndex {
    AAPLFragmentBufferIndexMipBias,
} AAPLFragmentBufferIndex;

typedef enum AAPLBlurTextureIndex {
    AAPLBlurTextureIndexInput = 0,
    AAPLBlurTextureIndexOutput = 1,
} AAPLBlurTextureIndex;

typedef enum AAPLFragmentTextureIndex {
    AAPLFragmentTextureIndexImage
} AAPLFragmentTextureIndex;

typedef struct AAPLVertex {
    vector_float2 position;
    vector_float2 texCoord;
} AAPLVertex;

#endif /* ShaderTypes_h */
