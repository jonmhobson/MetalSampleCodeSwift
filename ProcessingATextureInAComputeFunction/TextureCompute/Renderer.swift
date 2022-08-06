import Foundation
import MetalKit

final class Renderer: NSObject {
    private let vertices: MTLBuffer
    private let numVertices: Int
    private let inputTexture: MTLTexture
    private let outputTexture: MTLTexture

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let renderPipelineState: MTLRenderPipelineState
    private let computePipelineState: MTLComputePipelineState

    private var viewportSize: simd_uint2 = [0, 0]

    private let threadgroupSize: MTLSize
    private let threadgroupCount: MTLSize

    private static let triangleVertices: [AAPLVertex] = [
        AAPLVertex(position: [ 250, -250], textureCoordinate: [1, 1]),
        AAPLVertex(position: [-250, -250], textureCoordinate: [0, 1]),
        AAPLVertex(position: [-250,  250], textureCoordinate: [0, 0]),

        AAPLVertex(position: [ 250, -250], textureCoordinate: [1, 1]),
        AAPLVertex(position: [-250,  250], textureCoordinate: [0, 0]),
        AAPLVertex(position: [ 250,  250], textureCoordinate: [1, 0])
    ]

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue
        let textureLoader = MTKTextureLoader(device: device)

        // The original sample loads a TGA file manually. I think that distracts from the main point of the sample so
        // instead here we just load a texture from the Assets catalog using the MetalKit texture loader
        self.inputTexture = {
            do {
                return try textureLoader.newTexture(name: "Image", scaleFactor: metalView.window?.backingScaleFactor ?? 1.0, bundle: nil)
            } catch {
                fatalError("Could not load texture. Error: \(error)")
            }
        }()

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = inputTexture.pixelFormat
        textureDescriptor.width = inputTexture.width
        textureDescriptor.height = inputTexture.height
        textureDescriptor.usage = [.shaderWrite, .shaderRead]

        outputTexture = {
            guard let texture = device.makeTexture(descriptor: textureDescriptor) else { fatalError() }
            return texture
        }()

        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm_srgb

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let kernelFunction = defaultLibrary.makeFunction(name: "grayscaleKernel"),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        self.computePipelineState = {
            do {
                return try device.makeComputePipelineState(function: kernelFunction)
            } catch {
                fatalError("Failed to create compute pipeline state: \(error)")
            }
        }()

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "Simple Render Pipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.renderPipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        guard let buffer = device.makeBuffer(bytes: Renderer.triangleVertices,
                                     length: MemoryLayout<AAPLVertex>.stride * Renderer.triangleVertices.count,
                                             options: .storageModeShared) else {
            fatalError("Failed to create vertex buffer.")
        }

        vertices = buffer
        numVertices = Renderer.triangleVertices.count

        threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

        // Calculate the number of rows and columns of threadgroups given the size of the
        // input image. Ensure that the grid covers the entire image (or more).
        threadgroupCount = MTLSize(width: (inputTexture.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                   height: (inputTexture.height + threadgroupSize.height - 1) / threadgroupSize.height,
                                   depth: 1) // Image data is 2D, so set depth to 1.

        super.init()

        metalView.delegate = self
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = [UInt32(size.width), UInt32(size.height)]
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

        commandBuffer.label = "MyCommand"
        computeEncoder.label = "MyComputeEncoder"

        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setTexture(inputTexture, index: Int(AAPLTextureIndexInput.rawValue))
        computeEncoder.setTexture(outputTexture, index: Int(AAPLTextureIndexOutput.rawValue))
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        renderEncoder.label = "MyRenderEncoder"

        renderEncoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                              width: Double(viewportSize.x),
                                              height: Double(viewportSize.y),
                                              znear: 0, zfar: 0))

        renderEncoder.setRenderPipelineState(renderPipelineState)

        renderEncoder.setVertexBuffer(vertices, offset: 0, index: Int(AAPLVertexInputIndexVertices.rawValue))
        renderEncoder.setVertexBytes(&viewportSize,
                                     length: MemoryLayout<simd_uint2>.stride,
                                     index: Int(AAPLVertexInputIndexViewportSize.rawValue))
        renderEncoder.setFragmentTexture(outputTexture, index: Int(AAPLTextureIndexOutput.rawValue))
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: numVertices)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
