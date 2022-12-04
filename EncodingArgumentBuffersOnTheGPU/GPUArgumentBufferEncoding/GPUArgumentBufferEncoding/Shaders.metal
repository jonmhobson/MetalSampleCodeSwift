#include <metal_stdlib>
using namespace metal;

#import "ShaderTypes.h"

// Structure defining per instance parameters including textures
struct InstanceArguments {
    vector_float2       position;
    texture2d<float>    left_texture;
    texture2d<float>    right_texture;
};

// Structure defining an argument buffer holding a single texture.
// The source_textures parameter of the updateInstances kernel
// is an unbounded array of these structures.
struct SourceTextureArguments {
    texture2d<float>    texture;
};

// Every quad is an instance.
kernel void updateInstances(uint instanceID [[ thread_position_in_grid ]],
                            constant FrameState &frame_state [[ buffer(ComputeBufferIndexFrameState) ]],
                            const device SourceTextureArguments *source_textures [[ buffer(ComputeBufferIndexSourceTextures) ]],
                            device InstanceArguments *instance_params [[ buffer(ComputeBufferIndexInstanceParams) ]]) {
    // instanceID - The ID of the quad the kernel is currently updating.
    // frame_state - Info to calculate the position of the current quad.
    // source_texture - List of textures from which this kernel picks to apply to the quad
    // instance_params - An argument buffer array which stores the position of each quad and the textures
    //                   that should be applied to the quad for the current frame

    if (instanceID >= NumInstances) {
        return; // Return early if the instanceID is greater than the number of instances
    }

    // Calculate index of textures to write to argument buffers
    uint left_texture_index = (frame_state.textureIndexOffset + instanceID) % NumTextures;
    uint right_texture_index = (frame_state.textureIndexOffset + instanceID + 1) % NumTextures;

    // Calculate a position in clip space so that the quads line up nicely
    float2 gridPos = float2(instanceID % GridWidth, instanceID / GridWidth);
    float2 position = gridPos * float2(QuadSpacing) * frame_state.quadScale;
    position.x = -position.x;
    position += frame_state.offset;

    // Select the element in the instance_params array which stores the parameter for the quad.
    device InstanceArguments &quad_params = instance_params[instanceID];

    // Store the position of the quad.
    quad_params.position = position;

    // Select and store the textures to apply to this quad.
    quad_params.left_texture = source_textures[left_texture_index].texture;
    quad_params.right_texture = source_textures[right_texture_index].texture;
}

// Vertex shader outputs and per-fragment inputs
struct RasterizerData {
    float4  position [[position]];
    float2  tex_coord;
    uint    instanceID;
};

vertex RasterizerData
vertexShader(uint                           vertexID            [[ vertex_id ]],
             uint                           instanceID          [[ instance_id ]],
             const device Vertex            *vertices           [[ buffer(VertexBufferIndexVertices) ]],
             const device InstanceArguments *instance_params    [[ buffer(VertexBufferIndexInstanceParams) ]],
             constant FrameState            &frame_state        [[ buffer(VertexBufferIndexFrameState) ]]) {
    RasterizerData out;

    float2 quad_position = instance_params[instanceID].position;

    out.instanceID = instanceID;

    float2 position = vertices[vertexID].position * frame_state.quadScale + quad_position;

    out.position.xy = position;
    out.position.z = 0.0;
    out.position.w = 1.0;
    out.tex_coord = vertices[vertexID].texCoord;

    return out;
}

fragment float4
fragmentShader(RasterizerData in [[ stage_in ]],
               device InstanceArguments *instance_params [[ buffer(FragmentBufferIndexInstanceParams) ]],
               constant FrameState &frame_state [[ buffer(FragmentBufferIndexFrameState) ]]) {
    constexpr sampler texture_sampler (mag_filter::linear,
                                       min_filter::linear,
                                       mip_filter::linear);

    uint instanceID = in.instanceID;

    float4 output_color;

    // Choose the left sample as the output color for the left side of quad. Because
    // slideFactor increased from 0 to 1, this slides left texture in from the left side
    // of the quad.
    texture2d<float> left_texture = instance_params[instanceID].left_texture;
    texture2d<float> right_texture = instance_params[instanceID].right_texture;

    float4 left_sample = left_texture.sample(texture_sampler, in.tex_coord);
    float4 right_sample = right_texture.sample(texture_sampler, in.tex_coord);

    if (frame_state.slideFactor < in.tex_coord.x) {
        output_color = left_sample;
    } else {
        output_color = right_sample;
    }

    return output_color;
}
