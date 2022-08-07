import Foundation
import MetalKit

private func alignUp(_ inSize: Int, _ align: Int) -> Int {
    // Assert if align is not a power of 2
    assert(((align - 1) & align) == 0)
    let alignmentMask = align - 1
    return ((inSize + alignmentMask) & (~alignmentMask))
}

final class Renderer: NSObject {
    private let numImages = 6
    private var blurFrames = 0
    private let maxFramesPerImage = 300
    private var currentImageIndex = 0

    private let vertexBuffer: MTLBuffer

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderPipelineState: MTLRenderPipelineState


    private var imageTextures: [MTLTexture]
    private var displayTexture: MTLTexture? = nil
    private let imageHeap: MTLHeap
    private var scratchHeap: MTLHeap? = nil

    private let fence: MTLFence

    private let gaussianBlur: GaussianBlurFilter
    private let downsample: DownsampleFilter

    private var displayScale: SIMD2<Float> = .zero
    private var quadScale: SIMD2<Float> = .zero

    private static let vertexData: [AAPLVertex] = [
        AAPLVertex(position: [ 1, -1], texCoord: [1, 1]),
        AAPLVertex(position: [-1, -1], texCoord: [0, 1]),
        AAPLVertex(position: [-1,  1], texCoord: [0, 0]),

        AAPLVertex(position: [ 1, -1], texCoord: [1, 1]),
        AAPLVertex(position: [-1,  1], texCoord: [0, 0]),
        AAPLVertex(position: [ 1,  1], texCoord: [1, 0])
    ]

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm_srgb

        self.vertexBuffer = {
            guard let buffer = device.makeBuffer(bytes: Renderer.vertexData,
                                         length: MemoryLayout<AAPLVertex>.stride * Renderer.vertexData.count,
                                                 options: .storageModeShared) else {
                fatalError("Failed to create vertex buffer.")
            }
            return buffer
        }()

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "texturedQuadVertex"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "texturedQuadFragment") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Drawable Pipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat
        pipelineStateDescriptor.stencilAttachmentPixelFormat = metalView.depthStencilPixelFormat

        self.renderPipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        self.fence = {
            guard let fence = device.makeFence() else { fatalError() }
            return fence
        }()

        self.gaussianBlur = GaussianBlurFilter(device: device)
        self.downsample = DownsampleFilter(device: device)

        // MARK: loadImages
        let textureLoader = MTKTextureLoader(device: device)

        self.imageTextures = (0..<numImages).map { i in
            return try! textureLoader.newTexture(name: "Image\(i)", scaleFactor: 1.0, bundle: nil, options: [.SRGB: false])
        }

        // MARK: createImageHeap
        let heapDescriptor = MTLHeapDescriptor()

        heapDescriptor.storageMode = .private
        heapDescriptor.size = 0

        // Build a descriptor for each texture and calculate size needed to put the texture in the heap

        // This method of calculating the heap size is only guaranteed to be large enough for all the
        // textures if we also create the textures in the same order we're getting the sizeAndAlign
        // information. (i.e. if textures have different alignment requirements and we allocate in a
        // different order there may not be enough space for all textures)

        for imageTexture in imageTextures {
            // Create a descriptor using the texture's properties
            let descriptor = Renderer.newDescriptor(texture: imageTexture, storageMode: heapDescriptor.storageMode)

            // Determine the size needed for the heap from the given descriptor
            var sizeAndAlign = device.heapTextureSizeAndAlign(descriptor: descriptor)

            // Align the size so that more resources will fit after this texture
            sizeAndAlign.size = alignUp(sizeAndAlign.size, sizeAndAlign.align)

            // Accumulate the size required for the heap to hold this texture
            heapDescriptor.size += sizeAndAlign.size
        }

        // Create a heap large enough to hold all resources
        imageHeap = {
            guard let heap = device.makeHeap(descriptor: heapDescriptor) else { fatalError() }
            return heap
        }()

        // MARK: moveImagesToHeap
        // Create a command buffer and blit encoder to upload data from original resources to newly created
        // resources from the heap

        if let commandBuffer = commandQueue.makeCommandBuffer(), let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            commandBuffer.label = "Heap Upload Command Buffer"

            // Create new textures from the heap and copy contents of existing textures into the new textures
            for i in 0..<numImages {
                // Create descriptor using the texture's properties
                let descriptor = Renderer.newDescriptor(texture: imageTextures[i], storageMode: imageHeap.storageMode)

                // Create a texture from the heap
                let heapTexture = imageHeap.makeTexture(descriptor: descriptor)!

                // Blit every slice of every level from the original texture to the texture created from the heap
                var region = MTLRegionMake2D(0, 0, imageTextures[i].width, imageTextures[i].height)

                for level in 0..<imageTextures[i].mipmapLevelCount {
                    for slice in 0..<imageTextures[i].arrayLength {
                        blitEncoder.copy(from: imageTextures[i],
                                         sourceSlice: slice,
                                         sourceLevel: level,
                                         sourceOrigin: region.origin,
                                         sourceSize: region.size,
                                         to: heapTexture,
                                         destinationSlice: slice,
                                         destinationLevel: level,
                                         destinationOrigin: region.origin)
                    }

                    region.size.width /= 2
                    region.size.height /= 2
                    if region.size.width == 0 { region.size.width = 1 }
                    if region.size.height == 0 { region.size.height = 1 }
                }

                imageTextures[i] = heapTexture
            }

            blitEncoder.endEncoding()
            commandBuffer.commit()
        }

        super.init()

        metalView.delegate = self
    }

    static func newDescriptor(texture: MTLTexture, storageMode: MTLStorageMode) -> MTLTextureDescriptor {
        let descriptor = MTLTextureDescriptor()

        descriptor.textureType = texture.textureType
        descriptor.pixelFormat = texture.pixelFormat
        descriptor.width = texture.width
        descriptor.height = texture.height
        descriptor.depth = texture.depth
        descriptor.mipmapLevelCount = texture.mipmapLevelCount
        descriptor.arrayLength = texture.arrayLength
        descriptor.sampleCount = texture.sampleCount
        descriptor.storageMode = storageMode

        return descriptor
    }

    func createScratchHeap(inTexture: MTLTexture) {
        let heapStorageMode = MTLStorageMode.private

        let descriptor = Renderer.newDescriptor(texture: inTexture, storageMode: heapStorageMode)

        descriptor.storageMode = .private

        let downsampleSizeAndAlignRequirement = downsample.heapSizeAndAlignWithInputTextureDescriptor(inDescriptor: descriptor)
        let gaussianBlurSizeAndAlignRequirement = gaussianBlur.heapSizeAndAlignWithInputTextureDescriptor(inDescriptor: descriptor)

        let requiredAlignment = max(gaussianBlurSizeAndAlignRequirement.align, downsampleSizeAndAlignRequirement.align)
        let gaussianBlurSizeAligned = alignUp(gaussianBlurSizeAndAlignRequirement.size, requiredAlignment)
        let downsampleSizeAligned = alignUp(downsampleSizeAndAlignRequirement.size, requiredAlignment)
        let requiredSize = gaussianBlurSizeAligned + downsampleSizeAligned

        if scratchHeap == nil || requiredSize > scratchHeap!.maxAvailableSize(alignment: requiredAlignment) {
            let heapDesc = MTLHeapDescriptor()

            heapDesc.size = requiredSize
            heapDesc.storageMode = heapStorageMode

            scratchHeap = device.makeHeap(descriptor: heapDesc)
        }
    }

    func executeFilterGraph(inTexture: MTLTexture) -> MTLTexture {
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let scratchHeap = scratchHeap else { fatalError() }
        commandBuffer.label = "Filter Graph Commands"

        var resultTexture: MTLTexture? = inTexture

        resultTexture = downsample.execute(commandBuffer: commandBuffer, inTexture: inTexture, heap: scratchHeap, fence: fence)
        resultTexture = gaussianBlur.execute(commandBuffer: commandBuffer, inTexture: resultTexture!, heap: scratchHeap, fence: fence)

        commandBuffer.commit()
        return resultTexture!
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        if size.width < size.height {
            displayScale.x = 1.0
            displayScale.y = Float(size.width / size.height)
        } else {
            displayScale.x = Float(size.height / size.width)
            displayScale.y = 1.0
        }
    }

    func draw(in view: MTKView) {
        blurFrames += 1

        if displayTexture == nil || blurFrames > maxFramesPerImage {
            // Make memory of the display texture usable by new objects so that the filter operations
            // can temporarily use the memory until the renderer actually needs it
            displayTexture?.makeAliasable()

            let inTexture = imageTextures[currentImageIndex]

            createScratchHeap(inTexture: inTexture)

            displayTexture = executeFilterGraph(inTexture: inTexture)

            // Choose a new image to blur the next time
            currentImageIndex = (currentImageIndex + 1) % numImages

            // Restart the number of frames blurred
            blurFrames = 0
        }

        guard let displayTexture = displayTexture else { return }

        // Scale quad to maintain the image's aspect ratio and fit it within the display
        if displayTexture.width < displayTexture.height {
            quadScale.x = displayScale.x * Float(displayTexture.width) / Float(displayTexture.height)
            quadScale.y = displayScale.y
        } else {
            quadScale.x = displayScale.x
            quadScale.y = displayScale.y * Float(displayTexture.height) / Float(displayTexture.width)
        }

        // Create a new command buffer for each render pass to the current drawable
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Drawable Commands"

        // Obtain a render pass descriptor generated from the view's drawable textures
        if  let renderPassDescriptor = view.currentRenderPassDescriptor,
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {

            renderEncoder.label = "Drawable Render Encoder"
            renderEncoder.pushDebugGroup("DrawQuad")

            renderEncoder.setRenderPipelineState(renderPipelineState)

            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: Int(AAPLVertexBufferIndexVertices.rawValue))
            renderEncoder.setVertexBytes(&quadScale, length: MemoryLayout<SIMD2<Float>>.stride, index: Int(AAPLVertexBufferIndexScale.rawValue))

            renderEncoder.setFragmentTexture(displayTexture, index: 0)

            var lod = Float(blurFrames) / Float(maxFramesPerImage) * Float(displayTexture.mipmapLevelCount)
            renderEncoder.setFragmentBytes(&lod, length: MemoryLayout<Float>.stride, index: 0)

            // Wait for compute to finish before executing the fragment stage (which occurs during the next draw
            renderEncoder.waitForFence(fence, before: .fragment)

            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            renderEncoder.updateFence(fence, after: .fragment)

            renderEncoder.popDebugGroup()
            renderEncoder.endEncoding()

            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
        }

        commandBuffer.commit()
    }
}
