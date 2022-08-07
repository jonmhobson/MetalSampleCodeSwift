#include <metal_stdlib>
using namespace metal;

#import "ShaderTypes.h"

constant float gaussianWeights[5] { 0.06136, 0.24477, 0.38774, 0.24477, 0.06136 };

static void gaussianBlur(texture2d<half, access::read> inTexture,
                         texture2d<half, access::write> outTexture,
                         uint lod,
                         int2 offset,
                         uint2 gid) {
    uint2 textureDim(outTexture.get_width(), outTexture.get_height());
    if(all(gid < textureDim)) {
        half3 outColor(0.0);

        for(int i = -2; i < 3; ++i) {
            uint2 pixCoord = clamp(uint2(int2(gid) + offset * i), uint2(0), textureDim);
            outColor += inTexture.read(pixCoord, lod).rgb * gaussianWeights[i + 2];
        }

        outTexture.write(half4(outColor, 1.0), gid);
    }
}

kernel void gaussianBlurHorizontal(texture2d<half, access::read> inTexture [[texture(AAPLBlurTextureIndexInput)]],
                                   texture2d<half, access::write> outTexture [[texture(AAPLBlurTextureIndexOutput)]],
                                   constant uint &lod [[buffer(AAPLBlurBufferIndexLOD)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    gaussianBlur(inTexture, outTexture, lod, int2(1, 0), gid);
}

kernel void gaussianBlurVertical(texture2d<half, access::read> inTexture [[texture(AAPLBlurTextureIndexInput)]],
                                 texture2d<half, access::write> outTexture [[texture(AAPLBlurTextureIndexOutput)]],
                                 constant uint &lod [[buffer(AAPLBlurBufferIndexLOD)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    gaussianBlur(inTexture, outTexture, lod, int2(0, 1), gid);
}
