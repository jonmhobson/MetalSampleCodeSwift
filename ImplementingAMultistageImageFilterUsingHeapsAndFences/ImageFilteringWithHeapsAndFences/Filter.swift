import Metal

class Filter {
    let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
    }

    func heapSizeAndAlignWithInputTextureDescriptor(inDescriptor: MTLTextureDescriptor) -> MTLSizeAndAlign {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: inDescriptor.pixelFormat,
                                                                  width: inDescriptor.width,
                                                                  height: inDescriptor.height,
                                                                  mipmapped: true)

        return device.heapTextureSizeAndAlign(descriptor: descriptor)
    }
}

class GaussianBlurFilter: Filter {
    let horizontalKernel: MTLComputePipelineState
    let verticalKernel: MTLComputePipelineState

    override init(device: MTLDevice) {
        guard let defaultLibrary = device.makeDefaultLibrary(),
              let function = defaultLibrary.makeFunction(name: "gaussianBlurHorizontal"),
              let functionV = defaultLibrary.makeFunction(name: "gaussianBlurVertical") else { fatalError() }

        do {
            horizontalKernel = try device.makeComputePipelineState(function: function)
            verticalKernel = try device.makeComputePipelineState(function: functionV)
        } catch {
            fatalError("Failed creating a compute kernel: \(error)")
        }

        super.init(device: device)
    }

    override func heapSizeAndAlignWithInputTextureDescriptor(inDescriptor: MTLTextureDescriptor) -> MTLSizeAndAlign {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                         width: (inDescriptor.width >> 1),
                                                                         height: (inDescriptor.height >> 1),
                                                                         mipmapped: false)

        return device.heapTextureSizeAndAlign(descriptor: textureDescriptor)
    }

    func execute(commandBuffer: MTLCommandBuffer,
                 inTexture: MTLTexture,
                 heap: MTLHeap,
                 fence: MTLFence) -> MTLTexture {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 0, height: 0, mipmapped: false)

        // Heap resources must share the same storage mode as the heap.
        textureDescriptor.storageMode = heap.storageMode
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        for mipmapLevel in 1..<inTexture.mipmapLevelCount {
            textureDescriptor.width = inTexture.width >> mipmapLevel
            textureDescriptor.height = inTexture.height >> mipmapLevel

            if textureDescriptor.width <= 0 {
                textureDescriptor.width = 1
            }

            if textureDescriptor.height <= 0 {
                textureDescriptor.height = 1
            }

            guard let intermediaryTexture = heap.makeTexture(descriptor: textureDescriptor) else {
                fatalError("Failed to allocate on heap, did not request enough resources.")
            }

            let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupCount = MTLSize(width: (intermediaryTexture.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                           height: (intermediaryTexture.height + threadgroupSize.height - 1) / threadgroupSize.height,
                                           depth: 1)

            // Create a view of the input texture from the current mipmap level to output the final result
            let outTexture = inTexture.__newTextureView(with: inTexture.pixelFormat,
                                                       textureType: inTexture.textureType,
                                                       levels: NSMakeRange(mipmapLevel, 1),
                                                       slices: NSMakeRange(0, 1))

            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                // Wait for blit operation in DownsampleFilter AND operations from previous iterations
                // of this filter to complete before continuing
                computeEncoder.waitForFence(fence)

                computeEncoder.setComputePipelineState(horizontalKernel)
                computeEncoder.setTexture(inTexture, index: Int(AAPLBlurTextureIndexInput.rawValue))
                computeEncoder.setTexture(intermediaryTexture, index: Int(AAPLBlurTextureIndexOutput.rawValue))
                var mipmapLevel = mipmapLevel
                computeEncoder.setBytes(&mipmapLevel, length: MemoryLayout<Int>.stride, index: Int(AAPLBlurBufferIndexLOD.rawValue))

                computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

                // Perform vertical blur using the horizontally blurred texture as an input
                // and a view of the mipmap level of the input texture as the output
                computeEncoder.setComputePipelineState(verticalKernel)
                computeEncoder.setTexture(intermediaryTexture, index: Int(AAPLBlurTextureIndexInput.rawValue))
                computeEncoder.setTexture(outTexture, index: Int(AAPLBlurTextureIndexOutput.rawValue))

                var mipmapLevelZero = 0
                computeEncoder.setBytes(&mipmapLevelZero, length: MemoryLayout<Int>.stride, index: Int(AAPLBlurBufferIndexLOD.rawValue))
                computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)

                computeEncoder.updateFence(fence)

                computeEncoder.endEncoding()
            }

            intermediaryTexture.makeAliasable()
        }

        return inTexture
    }
}

class DownsampleFilter: Filter {
    func execute(commandBuffer: MTLCommandBuffer,
                 inTexture: MTLTexture,
                 heap: MTLHeap,
                 fence: MTLFence) -> MTLTexture {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: inTexture.pixelFormat,
                                                                         width: inTexture.width,
                                                                         height: inTexture.height,
                                                                         mipmapped: true)

        textureDescriptor.storageMode = heap.storageMode
        textureDescriptor.usage = [.shaderRead, .shaderWrite]

        guard let outTexture = heap.makeTexture(descriptor: textureDescriptor) else { fatalError() }

        if let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitCommandEncoder.copy(from: inTexture,
                                    sourceSlice: 0,
                                    sourceLevel: 0,
                                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                                    sourceSize: MTLSize(width: inTexture.width, height: inTexture.height, depth: inTexture.depth),
                                    to: outTexture,
                                    destinationSlice: 0,
                                    destinationLevel: 0,
                                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))

            blitCommandEncoder.generateMipmaps(for: outTexture)
            blitCommandEncoder.updateFence(fence)
            blitCommandEncoder.endEncoding()
        }

        return outTexture
    }
}
