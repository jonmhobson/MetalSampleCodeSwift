import Foundation
import MetalKit

let maxFramesInFlight = 3
let alignedUniformsSize = (MemoryLayout<Uniforms>.stride + 255) & ~255

final class Renderer: NSObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    let scene: Scene

    let uniformBuffer: MTLBuffer
    var resourceBuffer: MTLBuffer!

    var raytracingPipeline: MTLComputePipelineState!
    var copyPipeline: MTLRenderPipelineState!
    var intersectionFunctionTable: MTLIntersectionFunctionTable!

    var accumulationTargets: [MTLTexture] = []
    var randomTexture: MTLTexture!

    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []

    var instanceAccelerationStructure: MTLAccelerationStructure!
    var instanceBuffer: MTLBuffer!
    var resourcesStride: Int

    var useIntersectionFunctions = false

    var sem: DispatchSemaphore
    var size: CGSize = CGSize.zero
    var frameIndex = 0
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0

    init(device: MTLDevice, scene: Scene) {
        self.device = device
        self.scene = scene

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError()
        }

        self.commandQueue = commandQueue

        self.sem = DispatchSemaphore(value: maxFramesInFlight)

        guard let library = device.makeDefaultLibrary() else { fatalError() }
        self.library = library

        // MARK: createBuffers
        // The uniform buffer contains a few small values which change from frame to frame. The
        // sample can have up to 3 frames in flight at once, so allocate a range of the buffer
        // for each frame. The GPU reads from one chunk while the CPU writes to the next chunk.
        // Align the chunks to 256 bytes on macOS and 16 bytes on iOS.
        let uniformBufferSize = alignedUniformsSize * maxFramesInFlight
        let options = MTLResourceOptions.storageModeShared
        guard let uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: options) else {
            fatalError()
        }
        self.uniformBuffer = uniformBuffer

        scene.uploadToBuffers()

        resourcesStride = 0

        super.init()

        // Each intersection function has its own set of resources. Determine the maximum size over all
        // intersection functions. This will become the stride used by intersection functions to find
        // the starting address for their resources.
        for geometry in scene.geometries {
            let encoder = self.newArgumentEncoderForResources(resources: geometry.resources)
            if encoder.encodedLength > resourcesStride {
                resourcesStride = encoder.encodedLength
            }
        }

        resourceBuffer = device.makeBuffer(length: resourcesStride * scene.geometries.count, options: options)

        for (geometryIndex, geometry) in scene.geometries.enumerated() {
            // Create an argument encoder for this geometry's intersection function's resources
            let encoder = newArgumentEncoderForResources(resources: geometry.resources)

            // Bind the argument encoder to the resource buffer at this geometry's offset.
            encoder.setArgumentBuffer(resourceBuffer, offset: resourcesStride * geometryIndex)

            // Encode the arguments into the resource buffer.
            for (argumentIndex, resource) in geometry.resources.enumerated() {
                if resource.conforms(to: MTLBuffer.self) {
                    encoder.setBuffer((resource as! MTLBuffer), offset: 0, index: argumentIndex)
                } else if resource.conforms(to: MTLTexture.self) {
                    encoder.setTexture((resource as! MTLTexture), index: argumentIndex)
                }
            }
        }

        createAccelerationStructures()
        createPipelines()
    }

    private func specializedFunction(name: String) -> MTLFunction {
        // Fill out a dictionary of function constant values
        let constants = MTLFunctionConstantValues()

        // The first constant is the stride between entries in the resource buffer. The sample
        // uses this to allow intersection functions to look up any resources they use.
        var resourcesStride = resourcesStride
        constants.setConstantValue(&resourcesStride, type: .uint, index: 0)

        constants.setConstantValue(&useIntersectionFunctions, type: .bool, index: 1)

        // Finally, load the function from the Metal library.
        let function = try! library.makeFunction(name: name, constantValues: constants)

        return function
    }

    private func newComputePipelineState(function: MTLFunction, linkedFunctions: [MTLFunction]) -> MTLComputePipelineState {
        var mtlLinkedFunctions: MTLLinkedFunctions? = nil

        // Attach the additional functions to an MTLLinkedFunctions object
        if !linkedFunctions.isEmpty {
            mtlLinkedFunctions = MTLLinkedFunctions()
            mtlLinkedFunctions?.functions = linkedFunctions
        }

        let descriptor = MTLComputePipelineDescriptor()

        // Set the main compute function
        descriptor.computeFunction = function

        // Attach the linked functions object to the compute pipeline descriptor.
        descriptor.linkedFunctions = mtlLinkedFunctions

        // Set to YES to allow the compiler to make certain optimizations
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        // Create the compute pipeline state
        let pipeline = try! device.makeComputePipelineState(descriptor: descriptor, options: MTLPipelineOption(rawValue: 0), reflection: nil)

        return pipeline
    }

    private func createPipelines() {
        useIntersectionFunctions = false

        // Check if any scene geometry has an intersection function
        for geometry in scene.geometries {
            if geometry.intersectionFunctionName != nil {
                useIntersectionFunctions = true
                break
            }
        }

        // Maps intersection function names to actual MTLFunctions
        var intersectionFunctions: [String: MTLFunction] = [:]

        // First, load all the intersection functions since the sample needs them to create the final
        // ray-tracing compute pipeline state.
        for geometry in scene.geometries {
            // Skip the geometry if it doesn't have an intersection function or if the app already loaded it.
            if geometry.intersectionFunctionName == nil || intersectionFunctions[geometry.intersectionFunctionName!] != nil {
                continue
            }

            // Specialize function constants used by the intersection function
            let intersectionFunction = specializedFunction(name: geometry.intersectionFunctionName!)

            // Add the function to the dictionary
            intersectionFunctions[geometry.intersectionFunctionName!] = intersectionFunction
        }

        let raytracingFunction = specializedFunction(name: "raytracingKernel")

        // Create the compute pipeline state which does all of the ray tracing.
        raytracingPipeline = newComputePipelineState(function: raytracingFunction, linkedFunctions: Array(intersectionFunctions.values))

        // Create the intersection function table
        if useIntersectionFunctions {
            let intersectionFunctionTableDescriptor = MTLIntersectionFunctionTableDescriptor()

            intersectionFunctionTableDescriptor.functionCount = scene.geometries.count

            // Create a table large enough to hold all of the intersection functions. Metal
            // links intersection functions into the compute pipeline state, potentially with
            // a different address for each compute pipeline. Therefore, the intersection
            // function table is specific to the compute pipeline state that created it and you
            // can only use it with that pipeline.
            intersectionFunctionTable = raytracingPipeline.makeIntersectionFunctionTable(descriptor: intersectionFunctionTableDescriptor)

            // Bind the buffer used to pass resources to the intersection functions.
            intersectionFunctionTable.setBuffer(resourceBuffer, offset: 0, index: 0)

            // Map each piece of scene geometry to its intersection function.
            for (geometryIndex, geometry) in scene.geometries.enumerated() {
                if let ifName = geometry.intersectionFunctionName {
                    let intersectionFunction = intersectionFunctions[ifName]

                    // Create a handle to the copy of the intersection function linked into the
                    // ray-tracing compute pipeline state. Create a different handle for each pipeline
                    // it is linked with.
                    let handle = raytracingPipeline.functionHandle(function: intersectionFunction!)

                    // Insert the handle into the intersection function table. This ultimately maps the
                    // geometry's index to its intersection function.
                    intersectionFunctionTable.setFunction(handle, index: geometryIndex)
                }
            }
        }

        // Create a render pipeline state which copies the rendered scene into the MTKView and
        // performs simple tone mapping.
        let renderDescriptor = MTLRenderPipelineDescriptor()

        renderDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")

        renderDescriptor.colorAttachments[0].pixelFormat = .rgba16Float

        copyPipeline = try! device.makeRenderPipelineState(descriptor: renderDescriptor)
    }


    // Create acceleration structures for the scene. The scene contains primitive acceleration
    // structures and an instance acceleration structure. The primitive acceleration structures
    // contain primitives such as triangles and spheres. The instance acceleration structure contains
    // copies or "instances" of the primitive acceleartion structures, each with their own
    // transformation matrix describing where to place them in the scene.
    private func createAccelerationStructures() {
        let options = MTLResourceOptions.storageModeShared

        primitiveAccelerationStructures = []

        for (i, mesh) in scene.geometries.enumerated() {
            let geometryDescriptor = mesh.geometryDescriptor!

            // Assign each piece of geometry a consecutive slot in the intersection function table.
            geometryDescriptor.intersectionFunctionTableOffset = i

            // Create a primitive acceleration structure descriptor to contain the single piece
            // of acceleration structure geometry.
            let accelDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
            accelDescriptor.geometryDescriptors = [ geometryDescriptor ]

            // Build the acceleration structure.
            let accelerationStructure = self.newAccelerationStructureWithDescriptor(descriptor: accelDescriptor)

            // Add the acceleration structure to the array of primitive acceleration structures.
            primitiveAccelerationStructures.append(accelerationStructure)
        }

        // Allocate a buffer of acceleration structure instrance descriptors. Each descriptor represents
        // an instance of one of the primitive acceleration structures created above, with its own
        // transformation matrix.
        instanceBuffer = device.makeBuffer(length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * scene.instances.count, options: options)

        let instanceDescriptors = instanceBuffer.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: scene.instances.count)

        // Fill out instance descriptors.
        for (instanceIndex, instance) in scene.instances.enumerated() {
            let geometryIndex = scene.geometries.firstIndex {
                $0 == instance.geometry
            }!

            // Map the instance to its acceleration structure.
            instanceDescriptors[instanceIndex].accelerationStructureIndex = UInt32(geometryIndex)

            // Mark the instance as opaque if it doesn't have an intersection function so that the
            // ray intersector doesn't attempt to execute a function that doesn't exist.
            instanceDescriptors[instanceIndex].options = instance.geometry.intersectionFunctionName == nil ? .opaque : MTLAccelerationStructureInstanceOptions(rawValue: 0)

            // Metal adds the geometry intersection function table offset and instance intersection
            // function table offset together to determine which intersection function to execute.
            // The sample mapped geometries directly to their intersection functions above, so it
            // sets the instance's table offset to 0.
            instanceDescriptors[instanceIndex].intersectionFunctionTableOffset = 0

            // Set the instance mask, which the sample uses to filter out intersections between rays
            // and geometry. For example, it uses masks to prevent light sources from being visible
            // to secondary rays, which would result in their contribution being double-counted.
            instanceDescriptors[instanceIndex].mask = UInt32(instance.mask)

            // Copy the first three rows of the instance transformation matrix. Metal assumes that
            // the bottom row is (0, 0, 0, 1).
            // This allows instance descriptors to be tightly packed in memory.

            // We have to use keypaths here because MTLPackedFloat3 doesn't seem to have subscripts in swift
            let keyPathRows = [\MTLPackedFloat3.x, \MTLPackedFloat3.y, \MTLPackedFloat3.z]
            let keyPathColumns = [\MTLPackedFloat4x3.columns.0, \MTLPackedFloat4x3.columns.1, \MTLPackedFloat4x3.columns.2, \MTLPackedFloat4x3.columns.3]
            for (colIndex, column) in keyPathColumns.enumerated() {
                for (rowIndex, row) in keyPathRows.enumerated() {
                    instanceDescriptors[instanceIndex].transformationMatrix[keyPath: column][keyPath: row] = instance.transform[colIndex, rowIndex]
                }
            }

            // Create an instance acceleration structure descriptor.
            let accelDescriptor = MTLInstanceAccelerationStructureDescriptor()

            accelDescriptor.instancedAccelerationStructures = primitiveAccelerationStructures
            accelDescriptor.instanceCount = scene.instances.count
            accelDescriptor.instanceDescriptorBuffer = instanceBuffer

            // Finally, create the instance acceleration structure containing all of the instances
            // in the scene.
            instanceAccelerationStructure = newAccelerationStructureWithDescriptor(descriptor: accelDescriptor)
        }
    }

    // Create and compact an acceleration structure, given an acceleration structure descriptor.
    private func newAccelerationStructureWithDescriptor(descriptor: MTLAccelerationStructureDescriptor) -> MTLAccelerationStructure {
        // Query for the sizes needed to store and build the acceleration structure.
        let accelSizes = device.accelerationStructureSizes(descriptor: descriptor)

        // Allocate an acceleration structure large enough for this descriptor. This doesn't actually
        // build the acceleration structure, just allocates memory.
        let accelerationStructure = device.makeAccelerationStructure(size: accelSizes.accelerationStructureSize)!

        // Allocate scratch space used by Metal to build the acceleration structure.
        // Use MTLResourceStorageModePrivate for best performance since the sample
        // doesn't need access to buffer's contents.
        let scratchBuffer = device.makeBuffer(length: accelSizes.buildScratchBufferSize, options: .storageModePrivate)!

        // Create a command buffer which will perform the acceleration structure build
        var commandBuffer = commandQueue.makeCommandBuffer()!

        // Create an acceleration structure command encoder.
        var commandEncoder = commandBuffer.makeAccelerationStructureCommandEncoder()

        // Allocate a buffer for Metal to write the compacted accelerated structure's size into.
        let compactedSizeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        // Schedule the actual acceleration structure build
        commandEncoder?.build(accelerationStructure: accelerationStructure,
                              descriptor: descriptor,
                              scratchBuffer: scratchBuffer,
                              scratchBufferOffset: 0)

        // Compute and write the compacted acceleration structure size into the buffer. You
        // must already have a built accelerated structure because Metal determines the compacted
        // size based on the final size of the acceleration structure. Compacting an acceleration
        // structure can potentially reclaim significant amounts of memory since Metal must
        // create the initial structure using a conservative approach.
        commandEncoder?.writeCompactedSize(accelerationStructure: accelerationStructure,
                                           buffer: compactedSizeBuffer,
                                           offset: 0)

        // End encoding and commit the command buffer so the GPU can start building the
        // acceleration structure.
        commandEncoder?.endEncoding()

        commandBuffer.commit()

        // The sample waits for Metal to finish executing the command buffer so that it can
        // read back the compacted size.

        // Note: Don't wait for Metal to finish executing the comand buffer if you aren't compacting
        // the acceleration structure, as doing so requires CPU/GPU synchronization. You don't have
        // to compact acceleration structures, but you should when creating large static acceleration
        // structures, such as static scene geometry. Avoid compacting acceleration structures that
        // you rebuild every frame, as the synchronization cost may be significant.

        commandBuffer.waitUntilCompleted()

        let sizePointer = compactedSizeBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        let compactedSize = Int(sizePointer.pointee)

        // Allocate a smaller acceleration structure based on the returned size.
        let compactedAccelerationStructure = device.makeAccelerationStructure(size: compactedSize)!

        // Create another command buffer and encoder.
        commandBuffer = commandQueue.makeCommandBuffer()!

        commandEncoder = commandBuffer.makeAccelerationStructureCommandEncoder()

        // Encode the command to copy and compact the acceleration structure into the
        // smaller acceleration structure.
        commandEncoder?.copyAndCompact(sourceAccelerationStructure: accelerationStructure, destinationAccelerationStructure: compactedAccelerationStructure)

        // End encoding and commit the command buffer. You don't need to wait for Metal to finish
        // executing this command buffer as long as you synchronize any ray-intersection work
        // to run after this command buffer completes. The sample relies on Metal's default
        // dependency tracking on resources to automatically synchronize access to the new
        // compacted acceleration structure.
        commandEncoder?.endEncoding()
        commandBuffer.commit()

        return compactedAccelerationStructure
    }

    private func newArgumentEncoderForResources(resources: [any MTLResource]) -> MTLArgumentEncoder {
        var arguments: [MTLArgumentDescriptor] = []

        for resource in resources {
            let argumentDescriptor = MTLArgumentDescriptor()

            argumentDescriptor.index = arguments.count
            argumentDescriptor.access = .readOnly

            if resource.conforms(to: MTLBuffer.self) {
                argumentDescriptor.dataType = .pointer
            } else if let texture = resource as? MTLTexture {
                argumentDescriptor.dataType = .texture
                argumentDescriptor.textureType = texture.textureType
            }

            arguments.append(argumentDescriptor)
        }

        return device.makeArgumentEncoder(arguments: arguments)!
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        self.size = size

        // Create a pair of textures which the ray tracing kernel will use to accumulate
        // samples over several frames.
        let textureDescriptor = MTLTextureDescriptor()

        textureDescriptor.pixelFormat = .rgba32Float
        textureDescriptor.textureType = .type2D
        textureDescriptor.width = Int(size.width)
        textureDescriptor.height = Int(size.height)

        // Stored in private memory because only the GPU will read or write this texture.
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        accumulationTargets = []
        for _ in 0..<2 {
            accumulationTargets.append(device.makeTexture(descriptor: textureDescriptor)!)
        }

        textureDescriptor.pixelFormat = .r32Uint
        textureDescriptor.usage = .shaderRead

        // Create a texture containing a random integer value for each pixel. The sample
        // uses these values to decorrelate pixels while drawing pseudorandom numbers from the
        // Halton sequence.
        textureDescriptor.storageMode = .shared
        randomTexture = device.makeTexture(descriptor: textureDescriptor)

        // Initialize random values
        let numPixels = Int(size.width * size.height)
        let randomValues = Array(unsafeUninitializedCapacity: numPixels) { buffer, initializedCount in
            var i = 0
            var seed: UInt32 = 1
            // It's not allowed to use rand() in Swift, but this is the algorithm used.
            // The compiler recommends arc4random, but that's too slow.
            while i < numPixels {
                seed = seed &* 1103515245 &+ 12345
                buffer[i] = (seed / 65536) % 32768
                i+=1
            }
            initializedCount = numPixels
        }

        randomTexture.replace(region: MTLRegionMake2D(0, 0, Int(size.width), Int(size.height)),
                              mipmapLevel: 0,
                              withBytes: randomValues,
                              bytesPerRow: MemoryLayout<UInt32>.stride * Int(size.width))

        frameIndex = 0
    }

    private func updateUniforms() {
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        let uniforms = uniformBuffer.contents().advanced(by: uniformBufferOffset).bindMemory(to: Uniforms.self, capacity: 1)

        let position = scene.cameraPosition
        let target = scene.cameraTarget
        var up = scene.cameraUp

        let forward = simd_normalize(target - position)
        let right = simd_normalize(simd_cross(forward, up))
        up = simd_normalize(simd_cross(right, forward))

        uniforms[0].camera.position = position
        uniforms[0].camera.forward = forward
        uniforms[0].camera.right = right
        uniforms[0].camera.up = up

        let fieldOfView = 45.0 * (Float.pi / 180.0)
        let aspectRatio = Float(size.width / size.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight

        uniforms[0].camera.right *= imagePlaneWidth
        uniforms[0].camera.up *= imagePlaneHeight

        uniforms[0].width = UInt32(size.width)
        uniforms[0].height = UInt32(size.height)

        uniforms[0].frameIndex = UInt32(frameIndex)
        frameIndex += 1

        uniforms[0].lightCount = UInt32(scene.lights.count)

        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }

    func draw(in view: MTKView) {
        // The sample uses the uniform buffer to stream uniform data to the GPU, so it
        // needs to wait until the GPU finishes processing the oldest GPU frame before
        // it can reuse that space in the buffer
        sem.wait()

        // Create a command for the frame's commands.
        let commandBuffer = commandQueue.makeCommandBuffer()!

        // When the GPU finishes processing command buffer for the frame, signal the
        // semaphore to make the space in uniform avaialbe for future frames.

        let sem = self.sem

        // Note: Completion handlers should be as fast as possible as the GPU driver may
        // have other work scheduled on the underlying dispatch queue.
        commandBuffer.addCompletedHandler({ [sem] buffer in
            sem.signal()
        })

        updateUniforms()

        let width = Int(size.width)
        let height = Int(size.height)

        // Launch a rectangular grid of threads on the GPU to perform ray tracing, with one thread per
        // pixel. The sample needs to align the number of threads to a multiple of the threadgroup
        // size, because earlier, when it created the pipeline objects, it declared that the pipeline
        // would always use a threadgroup size that's a multiple of the thread execution width
        // (SIMD group size). An 8x8 threadgroup is a safe threadgroup size and small enough to be
        // supported on most devices. A more advanced app would choose the threadgroup size dynamically.
        let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
        let threadgroups = MTLSizeMake((width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                       (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                       1)

        // Create a compute encoder to encode GPU commands.
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

        // Bind buffers
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
        computeEncoder.setBuffer(resourceBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(instanceBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(scene.lightBuffer, offset: 0, index: 3)

        // Bind acceleration structure and intersection function table. These bind to normal buffer
        // binding slots.
        computeEncoder.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: 4)
        computeEncoder.setIntersectionFunctionTable(intersectionFunctionTable, bufferIndex: 5)

        // Bind textures. The ray tracing kernel reads from accumulationTargets[0], averages the
        // result with this frame's samples, and writes to accumulationTargets[1].
        computeEncoder.setTexture(randomTexture, index: 0)
        computeEncoder.setTexture(accumulationTargets[0], index: 1)
        computeEncoder.setTexture(accumulationTargets[1], index: 2)

        // Mark any resources used by intersection functions as "used". The sample does this because
        // it only references these resources indirectly via the resource buffer. Mteal makes all the
        // marked resources resident in memory before the intersection functions execute.
        // Normally, the sample would also mark the resource buffer iteself since the
        // intersection table references it indirectly. However, the sample also binds the resource
        // buffer directly, so it doesn't need to mark it explicity.
        for geometry in scene.geometries {
            for resource in geometry.resources {
                computeEncoder.useResource(resource, usage: .read)
            }
        }

        // Also mark primitive acceleration structures as used since only the instance acceleration
        // structure references them.
        for primitiveAccelerationStructure in primitiveAccelerationStructures {
            computeEncoder.useResource(primitiveAccelerationStructure, usage: .read)
        }

        // Bind the compute pipeline state.
        computeEncoder.setComputePipelineState(raytracingPipeline)

        // Dispatch the compute kernel to perform ray tracing.
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        // Swap the source and destination accumulation targets for the next frame
        accumulationTargets.swapAt(0, 1)

        if let currentDrawable = view.currentDrawable {
            // Copy the resulting image into the view using the graphics pipeline since the sample
            // can't write directly to it using the compute kernel. The sample delays getting the
            // current render pass descriptor as long as possible to avoid a lengthy stall waiting
            // for the CPU/compositor to release a drawable. The drawable may be nil if
            // the window moved off screen.
            let renderPassDescriptor = MTLRenderPassDescriptor()

            renderPassDescriptor.colorAttachments[0].texture = currentDrawable.texture
            renderPassDescriptor.colorAttachments[0].loadAction = .clear
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

            // Create a render command encoder.
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {

                renderEncoder.setRenderPipelineState(copyPipeline)
                renderEncoder.setFragmentTexture(accumulationTargets[0], index: 0)

                // Draw a quad which fills the screen.
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

                renderEncoder.endEncoding()

                commandBuffer.present(currentDrawable)
            }
        }

        commandBuffer.commit()
    }
}
