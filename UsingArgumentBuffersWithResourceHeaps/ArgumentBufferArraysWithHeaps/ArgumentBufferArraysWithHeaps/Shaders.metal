#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// Vertex shader outputs and per-fragment inputs.
struct RasterizerData {
    float4 position [[position]];
    float2 texCoord;
};

vertex RasterizerData vertexShader(uint vertexID [[vertex_id]],
                                   const device Vertex *vertices [[buffer(VertexBufferIndexVertices)]]) {
    RasterizerData out;

    float2 position = vertices[vertexID].position;

    out.position.xy = position;
    out.position.z = 0.0;
    out.position.w = 1.0;

    out.texCoord = vertices[vertexID].texCoord;

    return out;
}

struct FragmentShaderArguments {
    array<texture2d<float>, NumTextureArguments>    exampleTextures     [[ id(ArgumentBufferIDExampleTextures) ]];
    array<device float *, NumBufferArguments>       exampleBuffers      [[ id(ArgumentBufferIDExampleBuffers) ]];
    array<uint32_t, NumBufferArguments>             exampleConstants    [[ id(ArgumentBufferIDExampleConstants) ]];
};

fragment float4 fragmentShader(RasterizerData in [[stage_in]],
                               device FragmentShaderArguments &fragmentShaderArgs [[ buffer(FragmentBufferIndexArguments) ]]) {
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);

    float4 color = float4(0, 0, 0, 1);

    // If on the right side of the quad
    if (in.texCoord.x < 0.5) {
        // Use accumulated values from each of the 32 textures
        for (uint32_t textureToSample = 0; textureToSample < NumTextureArguments; textureToSample++) {
            float4 textureValue = fragmentShaderArgs.exampleTextures[textureToSample].sample(textureSampler, in.texCoord);
            color += textureValue;
        }
    } else { // On the left side of the quad
        // Use values from a buffer.

        // Use texCoord.x to select the buffer to read from
        uint32_t bufferToRead = (in.texCoord.x - 0.5) * 2.0 * (NumBufferArguments - 1);

        // Retrieve the number of elements for the selected buffer from
        // the array of constants in the argument buffer
        uint32_t numElements = fragmentShaderArgs.exampleConstants[bufferToRead];

        // Determine the index used to read from the buffer
        uint32_t indexToRead = in.texCoord.y * numElements;

        // Retrieve the buffer to read from by accessing the array of
        // buffers in the argument buffer
        device float* buffer = fragmentShaderArgs.exampleBuffers[bufferToRead];

        // Read from the buffer and assign the value to the output color
        color = buffer[indexToRead];
    }

    return color;
}
