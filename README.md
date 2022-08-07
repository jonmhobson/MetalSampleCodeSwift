# MetalSampleCodeSwift
Swift ports of Apple's Objective-C / C++ sample code

Metal is a great API, but it can feel inaccessible for Swift developers due to all the samples being written in C++ or Objective-C. 

Swift's more concise syntax and reduced boilerplate can also make the code a lot more readable.

Most game developers are coming from C++ so I understand why the samples are in C++ / Objective-C, but the goal of this project is to port all of the Metal samples so that App developers can also learn metal without such a large barrier to entry.

In the first pass only the macOS targets will be ported as the other targets add a lot of clutter, and the code is being tested on Apple Silicon and may not work on Intel.

| Samples ported so far  | Original sample |
| ------------- | ------------- |
| Performing Calculations on a GPU  | [link](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)  |
| Using Metal to Draw a View’s Contents | [link](https://developer.apple.com/documentation/metal/using_metal_to_draw_a_view_s_contents) |
| Using a Render Pipeline to Render Primitives | [link](https://developer.apple.com/documentation/metal/using_a_render_pipeline_to_render_primitives) |
| Synchronizing CPU and GPU Work | [link](https://developer.apple.com/documentation/metal/resource_synchronization/synchronizing_cpu_and_gpu_work) |
| Creating and Sampling Textures | [link](https://developer.apple.com/documentation/metal/textures/creating_and_sampling_textures) |
| Processing a Texture in a Compute Function | [link](https://developer.apple.com/documentation/metal/compute_passes/processing_a_texture_in_a_compute_function) |
| Calculating Primitive Visibility Using Depth Testing | [link](https://developer.apple.com/documentation/metal/render_passes/calculating_primitive_visibility_using_depth_testing) |
| Customizing Render Pass Setup | [link](https://developer.apple.com/documentation/metal/render_passes/customizing_render_pass_setup) |
| Encoding Indirect Command Buffers on the CPU | [link](https://developer.apple.com/documentation/metal/indirect_command_encoding/encoding_indirect_command_buffers_on_the_cpu) |
| Encoding Indirect Command Buffers on the GPU | [link](https://developer.apple.com/documentation/metal/indirect_command_encoding/encoding_indirect_command_buffers_on_the_gpu) |
| Managing groups of resources with argument buffers | [link](https://developer.apple.com/documentation/metal/buffers/managing_groups_of_resources_with_argument_buffers) |
| Adjusting the level of detail using Metal mesh shaders  | [link](https://developer.apple.com/documentation/metal/metal_sample_code_library/adjusting_the_level_of_detail_using_metal_mesh_shaders)  |
| Accelerating ray tracing using Metal  | [link](https://developer.apple.com/documentation/metal/metal_sample_code_library/accelerating_ray_tracing_using_metal)  |
| Implementing a Multistage Image Filter Using Heaps and Events | [link](https://developer.apple.com/documentation/metal/memory_heaps/implementing_a_multistage_image_filter_using_heaps_and_events) |
