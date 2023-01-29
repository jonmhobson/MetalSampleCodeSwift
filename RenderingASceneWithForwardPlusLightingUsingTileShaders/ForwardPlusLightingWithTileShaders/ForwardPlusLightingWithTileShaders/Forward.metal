#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "ShaderCommon.h"

struct ColorInOut {
    float4 position [[position]];
    float4 worldPos;
    float2 texCoord;
    half3 tangent;
    half3 bitangent;
    half3 normal;
};

vertex ColorInOut
forward_lighting_vertex(Vertex in [[stage_in]],
                        constant FrameData & frameData [[buffer(BufferIndexFrameData)]]) {
    ColorInOut out;
    float4 position = float4(in.position, 1.0);

    // Calculate the position in world space to reduce fragment-side math.
    out.worldPos = frameData.modelMatrix * position;

    // Calculate the position in clip space
    out.position = frameData.projectionMatrix * frameData.modelViewMatrix * position;

    // Set the UV pass-through values.
    out.texCoord = in.texCoord;

    // Rotate the tangents, bitangent and normal in eye-space.
    half3x3 normalMatrix = half3x3(frameData.normalMatrix);

    // Calculate the tanget, bitangent and normal in eye space.
    out.tangent = normalize(normalMatrix * in.tangent);
    out.bitangent = -normalize(normalMatrix * in.bitangent);
    out.normal = normalize(normalMatrix * in.normal);

    return out;
}

fragment half4
forward_lighting_fragment(ColorInOut in [[stage_in]],
                          constant FrameData & frameData [[buffer(BufferIndexFrameData)]],
                          threadgroup int * visible_lights [[threadgroup(ThreadgroupBufferIndexLightList)]],
                          threadgroup TileData * tile_data [[threadgroup(ThreadgroupBufferIndexTileData)]],
                          device PointLight * light_data [[buffer(BufferIndexLightsData)]],
                          device vector_float4 * light_positions [[buffer(BufferIndexLightsPosition)]],
                          texture2d<half> baseColorMap [[texture(TextureIndexBaseColor)]],
                          texture2d<half> normalMap [[texture(TextureIndexNormal)]],
                          texture2d<half> specularMap [[texture(TextureIndexSpecular)]]) {
    constexpr sampler linearSampler(mip_filter::linear,
                                    mag_filter::linear,
                                    min_filter::linear,
                                    s_address::repeat,
                                    t_address::repeat);

    int num_lights = min(atomic_load_explicit(&tile_data->numLights, memory_order_relaxed), MaxLightsPerTile);

    half4 normal_sample = normalMap.sample(linearSampler, in.texCoord.xy);
    half4 color_sample = baseColorMap.sample(linearSampler, in.texCoord.xy);
    half spec_contrib = specularMap.sample(linearSampler, in.texCoord.xy).r;

    // Move the normal from tangent space (texture space) to world space to perform lighting.
    half3 tangent_normal = normalize((normal_sample.xyz * 2.0) - 1.0);
    half3 eye_normal = normalize(tangent_normal.x * in.tangent +
                                 tangent_normal.y * in.bitangent +
                                 tangent_normal.z * in.normal);
    float4 world_space_normal = frameData.viewMatrixInv * float4(eye_normal.x,
                                                                 eye_normal.y,
                                                                 eye_normal.z, 0);

    // Calculate directional light contribution.
    float directional_light_intensity = max(
        (float)dot((float3)world_space_normal.xyz,
                   -frameData.directionalLightDirection),
        (float)0.0f
    );

    // Calculate pure directional light contribution (e.g. the sun).
    float4 out = float4(float3(color_sample.xyz) * frameData.directionalLightColor, 1) * directional_light_intensity;

    // Calculate ambient light contribution.
    out.xyz += float3(color_sample.xyz) * frameData.ambientLightColor;

    // Calculate the view vector for specular highlights.
    float4 cameraPos = frameData.viewMatrixInv[3];

    // Calculate the vector to the camera.
    float3 V = cameraPos.xyz - in.worldPos.xyz;

    // Apply point lights using the results of the culling tile shader.
    // Render only the lights that passed the visibility test for the current tile.
    for (int i = 0; i < num_lights; i++) {
        // Index into the visible light list.
        int lightId = visible_lights[i];

        // Obtain the light information for that index
        device PointLight &light = light_data[lightId];

        // Calculate attenuation.
        float3 to_light = light_positions[lightId].xyz - in.worldPos.xyz;
        float3 H = normalize(to_light + V);

        float length_sq = dot(to_light, to_light);
        to_light = normalize(to_light);

        float attenuation = fmax(1.0 - sqrt(length_sq) / light.lightRadius, 0.0);
        float diffuse = max(dot(float3(world_space_normal.xyz), float3(to_light)), 0.0);
        out.xyz += float3(light.lightColor.xyz) * float3(color_sample.xyz) * diffuse * attenuation;

        // Calculate specular contribution.
        out.xyz += powr(max(dot(float3(world_space_normal.xyz), H), 0.0), 32.0) * float3(color_sample.xyz)
            * spec_contrib * attenuation;
    }

    return half4(half3(out.xyz), 1.0f);
}
