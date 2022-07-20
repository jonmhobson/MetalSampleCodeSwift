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
| Adjusting the level of detail using Metal mesh shaders  | [link](https://developer.apple.com/documentation/metal/metal_sample_code_library/adjusting_the_level_of_detail_using_metal_mesh_shaders)  |
| Accelerating ray tracing using Metal  | [link](https://developer.apple.com/documentation/metal/metal_sample_code_library/accelerating_ray_tracing_using_metal)  |
