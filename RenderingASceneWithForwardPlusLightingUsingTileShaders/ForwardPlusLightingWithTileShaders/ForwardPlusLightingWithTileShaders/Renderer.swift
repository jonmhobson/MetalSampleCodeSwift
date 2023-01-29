import Foundation
import MetalKit

// Maximum number of command buffers in flight.
private let maxBuffersInFlight = 3

// The number of vertices in the fairy model.
private let numFairyVertices = 7
private let numLights = Int(NumLights)

private let depthDataPixelFormat = MTLPixelFormat.r32Float
private let depthBufferPixelFormat = MTLPixelFormat.depth32Float

private let threadgroupBufferSize = max(
    Int(MaxLightsPerTile) * MemoryLayout<UInt32>.stride,
    Int(TileWidth * TileHeight) * MemoryLayout<UInt32>.stride
)

extension BufferIndices {
    var index: Int {
        return Int(self.rawValue)
    }
}

extension VertexAttributes {
    var index: Int {
        return Int(self.rawValue)
    }
}

extension RenderTargetIndices {
    var index: Int {
        return Int(self.rawValue)
    }
}

extension TextureIndices {
    var index: Int {
        return Int(self.rawValue)
    }
}

extension ThreadgroupIndices {
    var index: Int {
        return Int(self.rawValue)
    }
}

final class Renderer: NSObject {
    private let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Vertex descriptor for models loaded with MetalKit and used for render pipelines.
    private let defaultVertexDescriptor: MTLVertexDescriptor

    // Pipeline state objects
    let depthPrePassPipelineState: MTLRenderPipelineState
    let lightBinCreationPipelineState: MTLRenderPipelineState
    let lightCullingPipelineState: MTLRenderPipelineState
    let forwardLightingPipelineState: MTLRenderPipelineState
    let fairyPipelineState: MTLRenderPipelineState

    // Depth state objects
    let defaultDepthState: MTLDepthStencilState
    let relaxedDepthStencilState: MTLDepthStencilState
    let dontWriteDepthState: MTLDepthStencilState

    // Buffers used to store dynamically-changing per-frame data.
    var frameDataBuffers: [MTLBuffer] = []

    // Buffers used to store dynamically-changing light positions.
    var lightWorldPositions: [MTLBuffer] = []
    var lightEyePositions: [MTLBuffer] = []

    var drawableSize: CGSize = .zero

    // Current buffer to fill with per frame data and set for the current frame.
    var currentBufferIndex: Int = 0

    // Buffer for constant light data.
    let lightsData: MTLBuffer

    // Field of view used to create perspective projection (in radians).
    var fov: Float = 1.0

    // Near-depth plane for the projection
    var nearPlane: Float = 0.0

    // Far-depth plane for the projection matrix
    var farPlane: Float = 0.0

    // Projection matrix calculated as a function of the view size.
    var projectionMatrix: matrix_float4x4 = .identity

    // Current rotation of the object (in radians).
    var rotation: Float = 0.0

    // Array of app-specific mesh objects in the scene.
    var meshes: [Mesh]

    // Custom render pass descriptor used to render to the view's drawable
    let viewRenderPassDescriptor: MTLRenderPassDescriptor

    // Mesh buffer for fairies.
    let fairy: MTLBuffer

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        metalView.device = device
        metalView.preferredFramesPerSecond = 120
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        // MARK: loadMetalWithMetalKitView

        // Create and load the basic Metal state objects

        // Load all the shader files with a .metal file extension in the project.
        let defaultLibrary = device.makeDefaultLibrary()!

        // Create an allocate frame data buffer objects.
        for i in 0..<maxBuffersInFlight {
            // Indicate shared storage so that both the CPU and the GPU can access the buffers.
            let storageMode = MTLResourceOptions.storageModeShared

            frameDataBuffers.append({
                let buffer = device.makeBuffer(length: MemoryLayout<FrameData>.stride, options: storageMode)!
                buffer.label = "FrameDataBuffer\(i)"
                return buffer
            }())

            lightWorldPositions.append({
                let buffer = device.makeBuffer(length: MemoryLayout<vector_float4>.stride * Int(NumLights), options: storageMode)!
                buffer.label = "LightPositions\(i)"
                return buffer
            }())

            lightEyePositions.append({
                let buffer = device.makeBuffer(length: MemoryLayout<vector_float4>.stride * Int(NumLights), options: storageMode)!
                buffer.label = "LightPositionsEyeSpace\(i)"
                return buffer
            }())
        }

        // Create a vertex descriptor for the Metal pipeline. Specify the layout of vertices the
        // pipeline should expect. The layout below keeps attributes used to calculate vertex shader
        // output position (world position, skinning, tweening weights) separate from other
        // attributes (texture coordinates, normals). This generally maximizes pipeline efficiency.
        defaultVertexDescriptor = MTLVertexDescriptor()

        // Positions.
        defaultVertexDescriptor.attributes[VertexAttributePosition.index].format = .float3
        defaultVertexDescriptor.attributes[VertexAttributePosition.index].offset = 0
        defaultVertexDescriptor.attributes[VertexAttributePosition.index].bufferIndex = BufferIndexMeshPositions.index

        // Texture coordinates.
        defaultVertexDescriptor.attributes[VertexAttributeTexcoord.index].format = .float2
        defaultVertexDescriptor.attributes[VertexAttributeTexcoord.index].offset = 0
        defaultVertexDescriptor.attributes[VertexAttributeTexcoord.index].bufferIndex = BufferIndexMeshGenerics.index

        // Normals.
        defaultVertexDescriptor.attributes[VertexAttributeNormal.index].format = .half4
        defaultVertexDescriptor.attributes[VertexAttributeNormal.index].offset = 8
        defaultVertexDescriptor.attributes[VertexAttributeNormal.index].bufferIndex = BufferIndexMeshGenerics.index

        // Tangents.
        defaultVertexDescriptor.attributes[VertexAttributeTangent.index].format = .half4
        defaultVertexDescriptor.attributes[VertexAttributeTangent.index].offset = 16
        defaultVertexDescriptor.attributes[VertexAttributeTangent.index].bufferIndex = BufferIndexMeshGenerics.index

        // Bitangents.
        defaultVertexDescriptor.attributes[VertexAttributeBitangent.index].format = .half4
        defaultVertexDescriptor.attributes[VertexAttributeBitangent.index].offset = 24
        defaultVertexDescriptor.attributes[VertexAttributeBitangent.index].bufferIndex = BufferIndexMeshGenerics.index

        // Position buffer layout.
        defaultVertexDescriptor.layouts[BufferIndexMeshPositions.index].stride = 12
        defaultVertexDescriptor.layouts[BufferIndexMeshPositions.index].stepRate = 1
        defaultVertexDescriptor.layouts[BufferIndexMeshPositions.index].stepFunction = .perVertex

        // Generic attribute buffer layout.
        defaultVertexDescriptor.layouts[BufferIndexMeshGenerics.index].stride = 32
        defaultVertexDescriptor.layouts[BufferIndexMeshGenerics.index].stepRate = 1
        defaultVertexDescriptor.layouts[BufferIndexMeshGenerics.index].stepFunction = .perVertex

        metalView.colorPixelFormat = .bgra8Unorm_srgb

        // Set view's depth stencil pixel format to Invalid. This app will manually manage its own
        // depth buffer, not depend on the depth buffer managed by MTKView
        metalView.depthStencilPixelFormat = .invalid

        // Create a render pipeline state descriptor
        let renderPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineStateDescriptor.rasterSampleCount = Int(NumSamples)
        renderPipelineStateDescriptor.vertexDescriptor = defaultVertexDescriptor

        renderPipelineStateDescriptor.colorAttachments[RenderTargetLighting.index].pixelFormat = metalView.colorPixelFormat
        renderPipelineStateDescriptor.colorAttachments[RenderTargetDepth.index].pixelFormat = depthDataPixelFormat

        renderPipelineStateDescriptor.depthAttachmentPixelFormat = depthBufferPixelFormat

        // Set unique descriptor values for the depth pre-pass pipeline state.
        let depthPrePassVertexFunction = defaultLibrary.makeFunction(name: "depth_pre_pass_vertex")!
        let depthPrePassFragmentFunction = defaultLibrary.makeFunction(name: "depth_pre_pass_fragment")!
        renderPipelineStateDescriptor.label = "Depth Pre-Pass"
        renderPipelineStateDescriptor.vertexDescriptor = defaultVertexDescriptor
        renderPipelineStateDescriptor.vertexFunction = depthPrePassVertexFunction
        renderPipelineStateDescriptor.fragmentFunction = depthPrePassFragmentFunction
        depthPrePassPipelineState = try! device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)

        // Set unique descriptor values for the standard matieral pipeline state.
        let vertexStandardMaterial = defaultLibrary.makeFunction(name: "forward_lighting_vertex")!
        let fragmentStandardMaterial = defaultLibrary.makeFunction(name: "forward_lighting_fragment")!
        renderPipelineStateDescriptor.label = "Forward Lighting"
        renderPipelineStateDescriptor.vertexDescriptor = defaultVertexDescriptor
        renderPipelineStateDescriptor.vertexFunction = vertexStandardMaterial
        renderPipelineStateDescriptor.fragmentFunction = fragmentStandardMaterial
        forwardLightingPipelineState = try! device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)

        // Set unique descriptor values for the fairy pipeline state.
        let fairyVertexFunction = defaultLibrary.makeFunction(name: "fairy_vertex")!
        let fairyFragmentFunction = defaultLibrary.makeFunction(name: "fairy_fragment")!
        renderPipelineStateDescriptor.label = "Fairy"
        renderPipelineStateDescriptor.vertexDescriptor = nil
        renderPipelineStateDescriptor.vertexFunction = fairyVertexFunction
        renderPipelineStateDescriptor.fragmentFunction = fairyFragmentFunction
        fairyPipelineState = try! device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)

        let binCreationKernel = defaultLibrary.makeFunction(name: "create_bins")!
        let binCreationPipelineDescriptor = MTLTileRenderPipelineDescriptor()
        binCreationPipelineDescriptor.label = "Light Bin Creation"
        binCreationPipelineDescriptor.rasterSampleCount = Int(NumSamples)
        binCreationPipelineDescriptor.colorAttachments[RenderTargetLighting.index].pixelFormat = metalView.colorPixelFormat
        binCreationPipelineDescriptor.colorAttachments[RenderTargetDepth.index].pixelFormat = depthDataPixelFormat
        binCreationPipelineDescriptor.threadgroupSizeMatchesTileSize = true
        binCreationPipelineDescriptor.tileFunction = binCreationKernel
        lightBinCreationPipelineState = try! device.makeRenderPipelineState(tileDescriptor: binCreationPipelineDescriptor, options: [], reflection: nil)

        // Create a tile render pipeline state descriptor for the culling pipeline state.
        let tileRenderPipelineDescriptor = MTLTileRenderPipelineDescriptor()
        let lightCullingKernel = defaultLibrary.makeFunction(name: "cull_lights")!
        tileRenderPipelineDescriptor.tileFunction = lightCullingKernel
        tileRenderPipelineDescriptor.label = "Light Culling"
        tileRenderPipelineDescriptor.rasterSampleCount = Int(NumSamples)

        tileRenderPipelineDescriptor.colorAttachments[RenderTargetLighting.index].pixelFormat = metalView.colorPixelFormat
        tileRenderPipelineDescriptor.colorAttachments[RenderTargetDepth.index].pixelFormat = depthDataPixelFormat

        tileRenderPipelineDescriptor.threadgroupSizeMatchesTileSize = true
        lightCullingPipelineState = try! device.makeRenderPipelineState(tileDescriptor: tileRenderPipelineDescriptor, options: [], reflection: nil)

        let depthStateDesc = MTLDepthStencilDescriptor()

        // Create a depth state with depth buffer write enabled.
        // Use .less because you render on a clean depth buffer
        depthStateDesc.depthCompareFunction = .less
        depthStateDesc.isDepthWriteEnabled = true
        defaultDepthState = device.makeDepthStencilState(descriptor: depthStateDesc)!

        // Create a depth state with depth buffer write disabled and set the comparison function to .lessEqual
        // The comparison function is .lessEqual instead of .less
        // The geometry pass renders to a pre-populated depth buffer (depth pre-pass) so each
        // fragment needs to pass if its z-value is equal to the existing value already in the depth buffer.
        depthStateDesc.depthCompareFunction = .lessEqual
        depthStateDesc.isDepthWriteEnabled = false
        relaxedDepthStencilState = device.makeDepthStencilState(descriptor: depthStateDesc)!

        // Create a depth state with depth buffer write disabled.
        depthStateDesc.depthCompareFunction = .lessEqual
        depthStateDesc.isDepthWriteEnabled = false
        dontWriteDepthState = device.makeDepthStencilState(descriptor: depthStateDesc)!

        // Create a render pass descriptor to render to the drawable
        viewRenderPassDescriptor = MTLRenderPassDescriptor()
        viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].loadAction = .clear
        viewRenderPassDescriptor.colorAttachments[RenderTargetDepth.index].loadAction = .clear
        viewRenderPassDescriptor.colorAttachments[RenderTargetDepth.index].storeAction = .dontCare
        viewRenderPassDescriptor.depthAttachment.loadAction = .clear
        viewRenderPassDescriptor.depthAttachment.storeAction = .dontCare
        viewRenderPassDescriptor.stencilAttachment.loadAction = .clear
        viewRenderPassDescriptor.stencilAttachment.storeAction = .dontCare
        viewRenderPassDescriptor.depthAttachment.clearDepth = 1.0
        viewRenderPassDescriptor.stencilAttachment.clearStencil = 0

        viewRenderPassDescriptor.tileWidth = Int(TileWidth)
        viewRenderPassDescriptor.tileHeight = Int(TileHeight)
        viewRenderPassDescriptor.threadgroupMemoryLength = threadgroupBufferSize + Int(TileDataSize)

        //        viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].storeAction =
        //            (NumSamples > 1) ? .multisampleResolve : .store
        viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].storeAction = .multisampleResolve

        // MARK: loadAssets
        // Create and load assets into Metal objects.

        // Create a Model I/O vertex descriptor so that the format and layout of Model I/O mesh vertices
        // fits the Metal render pipeline's vertex descriptor layout.
        let modelIOVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(defaultVertexDescriptor)

        // Indicate how each Metal vertex descriptor attribute maps to each Model I/O attribute
        (modelIOVertexDescriptor.attributes[VertexAttributePosition.index] as! MDLVertexAttribute).name = MDLVertexAttributePosition
        (modelIOVertexDescriptor.attributes[VertexAttributeTexcoord.index] as! MDLVertexAttribute).name = MDLVertexAttributeTextureCoordinate
        (modelIOVertexDescriptor.attributes[VertexAttributeNormal.index] as! MDLVertexAttribute).name = MDLVertexAttributeNormal
        (modelIOVertexDescriptor.attributes[VertexAttributeTangent.index] as! MDLVertexAttribute).name = MDLVertexAttributeTangent
        (modelIOVertexDescriptor.attributes[VertexAttributeBitangent.index] as! MDLVertexAttribute).name = MDLVertexAttributeBitangent


        guard let modelFileURL = Bundle.main.url(forResource: "Meshes/Temple.obj", withExtension: nil) else { fatalError("Could not find model in Bundle") }

        meshes = Mesh.newMeshes(url: modelFileURL,
                                vertexDescriptor: modelIOVertexDescriptor,
                                device: device)

        lightsData = device.makeBuffer(length: MemoryLayout<PointLight>.stride * Int(NumLights))!
        lightsData.label = "LightData"

        // Create a simple 2D triangle strip circle mesh for the fairies
        let fairySize: Float = 2.5
        var fairyVertices: [SimpleVertex] = []

        let angle = 2.0 * Float.pi / Float(numFairyVertices)

        for vtx in 0..<numFairyVertices {
            let point = Float((vtx % 2 != 0) ? (vtx + 1) / 2 : -vtx / 2)
            let position = vector_float2(x: sinf(point * angle), y: cosf(point * angle)) * fairySize
            fairyVertices.append(SimpleVertex(position: position))
        }

        fairy = device.makeBuffer(bytes: &fairyVertices, length: MemoryLayout<SimpleVertex>.stride * numFairyVertices)!

        // MARK: populateLights
        var lightData = lightsData.contents().bindMemory(to: PointLight.self, capacity: numFairyVertices)
        var lightPosition = lightWorldPositions[0].contents().bindMemory(to: vector_float4.self, capacity: numFairyVertices)


        for lightId in 0..<numLights {
            var distance: Float = 0
            var height: Float = 0
            var angle: Float = 0

            if lightId < numLights / 4 {
                distance = Float.random(in: 140...260)
                height = Float.random(in: 140...150)
                angle = Float.random(in: 0...(.pi * 2.0))
            } else if lightId < (numLights * 3) / 4 {
                distance = Float.random(in: 350...362)
                height = Float.random(in: 140...400)
                angle = Float.random(in: 0...(.pi * 2.0))
            } else if lightId < (numLights * 15) / 16 {
                distance = Float.random(in: 400...480)
                height = Float.random(in: 68...80)
                angle = Float.random(in: 0...(.pi * 2.0))
            } else {
                distance = Float.random(in: 40...40)
                height = Float.random(in: 220...350)
                angle = Float.random(in: 0...(.pi * 2.0))
            }

            lightData.pointee.lightRadius = Float.random(in: 25...35)
            lightPosition.pointee = vector_float4(distance * sinf(angle),
                                                  height,
                                                  distance * cosf(angle),
                                                  lightData.pointee.lightRadius)
            lightData.pointee.lightSpeed = Float.random(in: 0.003...0.015)
            let colorId = Int.random(in: 0..<3)
            if colorId == 0 {
                lightData.pointee.lightColor = vector_float3(Float.random(in: 2...3),
                                                             Float.random(in: 0...2),
                                                             Float.random(in: 0...2))
            } else if colorId == 1 {
                lightData.pointee.lightColor = vector_float3(Float.random(in: 0...2),
                                                             Float.random(in: 2...3),
                                                             Float.random(in: 0...2))
            } else {
                lightData.pointee.lightColor = vector_float3(Float.random(in: 0...2),
                                                             Float.random(in: 0...2),
                                                             Float.random(in: 2...3))
            }

            lightData = lightData.advanced(by: 1)
            lightPosition = lightPosition.advanced(by: 1)
        }

        memcpy(lightWorldPositions[1].contents(), lightWorldPositions[0].contents(), numLights * MemoryLayout<vector_float3>.stride)
        memcpy(lightWorldPositions[2].contents(), lightWorldPositions[0].contents(), numLights * MemoryLayout<vector_float3>.stride)

        super.init()

        metalView.delegate = self
    }

    func updateLights() {
        let previousFramesBufferIndex = (currentBufferIndex + maxBuffersInFlight - 1) % maxBuffersInFlight

        let lightData = lightsData.contents().bindMemory(to: PointLight.self, capacity: numLights)
        let frameData = frameDataBuffers[currentBufferIndex].contents().bindMemory(to: FrameData.self, capacity: 1)

        let viewMatrix = frameData.pointee.viewMatrix

        let previousWorldSpacePositions = lightWorldPositions[previousFramesBufferIndex].contents().bindMemory(to: vector_float4.self, capacity: numLights)

        let currentWorldSpaceLightPositions = lightWorldPositions[currentBufferIndex].contents().bindMemory(to: vector_float4.self, capacity: numLights)

        let currentEyeSpaceLightPositions = lightEyePositions[currentBufferIndex].contents().bindMemory(to: vector_float4.self, capacity: numLights)

        for i in 0..<numLights {
            let rotation = matrix_float4x4.rotation(lightData[i].lightSpeed, [0, 1, 0])

            let previousWorldSpacePosition = vector_float4(previousWorldSpacePositions[i].x,
                                                           previousWorldSpacePositions[i].y,
                                                           previousWorldSpacePositions[i].z, 1)

            var currentWorldSpacePosition = matrix_multiply(rotation, previousWorldSpacePosition)
            var currentEyeSpacePosition = matrix_multiply(viewMatrix, currentWorldSpacePosition)

            currentWorldSpacePosition.w = lightData[i].lightRadius
            currentEyeSpacePosition.w = lightData[i].lightRadius

            currentWorldSpaceLightPositions[i] = currentWorldSpacePosition
            currentEyeSpaceLightPositions[i] = currentEyeSpacePosition
        }
    }

    func updateFrameState() {
        currentBufferIndex = (currentBufferIndex + 1) % maxBuffersInFlight

        let frameData = frameDataBuffers[currentBufferIndex].contents().bindMemory(to: FrameData.self, capacity: 1)

        // Update ambient light color
        let ambientLightColor = vector_float3(0.05, 0.05, 0.05)
        frameData.pointee.ambientLightColor = ambientLightColor

        // Update directional light direction in world space.
        let directionalLightDirection = vector_float3(1.0, -1.0, 1.0)
        frameData.pointee.directionalLightDirection = directionalLightDirection

        // Update directional light color.
        let directionalLightColor = vector_float3(0.4, 0.4, 0.4)
        frameData.pointee.directionalLightColor = directionalLightColor

        // Set projection matrix and calculate inverted projection matrix.
        frameData.pointee.projectionMatrix = projectionMatrix
        frameData.pointee.projectionMatrixInv = projectionMatrix.inverse
        frameData.pointee.depthUnproject = vector_float2(farPlane / (farPlane - nearPlane),
            (-farPlane * nearPlane) / (farPlane - nearPlane))

        // Set screen dimensions.
        frameData.pointee.framebufferWidth = uint(drawableSize.width)
        frameData.pointee.framebufferHeight = uint(drawableSize.height)

        let fovScale = tanf(0.5 * fov) * 2.0
        let aspectRatio = Float(frameData.pointee.framebufferWidth) / Float(frameData.pointee.framebufferHeight)
        frameData.pointee.screenToViewSpace =
        vector_float3(
            fovScale / Float(frameData.pointee.framebufferHeight),
            -fovScale * 0.5 * aspectRatio,
            -fovScale * 0.5
        )

        // Calculate new view matrix and inverted view matrix.
        frameData.pointee.viewMatrix = matrix_multiply(simd_float4x4.translation([0.0, -75, 1000.5]), matrix_multiply(simd_float4x4.rotation(-0.5, [1, 0, 0]), simd_float4x4.rotation(rotation, [0, 1, 0])))

        frameData.pointee.viewMatrixInv = frameData.pointee.viewMatrix.inverse

        let rotationAxis = vector_float3(0, 1, 0)
        var modelMatrix = simd_float4x4.rotation(0, rotationAxis)
        let translation = simd_float4x4.translation(.zero)
        modelMatrix = matrix_multiply(modelMatrix, translation)

        frameData.pointee.modelViewMatrix = matrix_multiply(frameData.pointee.viewMatrix, modelMatrix)
        frameData.pointee.modelMatrix = modelMatrix

        frameData.pointee.normalMatrix = modelMatrix.upperLeft.transpose.inverse

        rotation += 0.002

        updateLights()
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size

        // Update the aspect ratio and projection matrix because the view orientation
        // or size has changed.
        let aspect = Float(size.width) / Float(size.height)
        fov = 65.0 * (Float.pi / 180.0)
        nearPlane = 1.0
        farPlane = 1500.0
        projectionMatrix = simd_float4x4.perspectiveLeftHand(fovyRadians: fov, aspect: aspect, nearZ: nearPlane, farZ: farPlane)

        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.width = Int(size.width)
        textureDescriptor.height = Int(size.height)
        textureDescriptor.usage = .renderTarget
        textureDescriptor.storageMode = .memoryless

        if NumSamples > 1 {
            textureDescriptor.sampleCount = Int(NumSamples)
            textureDescriptor.textureType = .type2DMultisample
            textureDescriptor.pixelFormat = view.colorPixelFormat

            let msaaColorTexture = device.makeTexture(descriptor: textureDescriptor)

            viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].texture = msaaColorTexture
        } /*else {
            textureDescriptor.textureType = .type2D
        }*/

        // Create depth buffer texture for depth testing
        textureDescriptor.pixelFormat = depthBufferPixelFormat
        let depthBufferTexture = device.makeTexture(descriptor: textureDescriptor)
        viewRenderPassDescriptor.depthAttachment.texture = depthBufferTexture

        // Create depth data texture to determine min max depth for each tile
        textureDescriptor.pixelFormat = depthDataPixelFormat
        let depthDataTexture = device.makeTexture(descriptor: textureDescriptor)
        viewRenderPassDescriptor.colorAttachments[RenderTargetDepth.index].texture = depthDataTexture
    }

    private func drawMeshes(renderEncoder: MTLRenderCommandEncoder) {
        for mesh in meshes {
            let metalKitMesh = mesh.metalKitMesh

            // Set the mesh's vertex buffers
            for bufferIndex in 0..<metalKitMesh.vertexBuffers.count {
                let vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex]
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: bufferIndex)
            }

            // Draw each submesh of the mesh.
            for submesh in mesh.submeshes {
                // Set any textures that you read or sample in the render pipeline.
                renderEncoder.setFragmentTexture(submesh.textures[TextureIndexBaseColor.index], index: TextureIndexBaseColor.index)
                renderEncoder.setFragmentTexture(submesh.textures[TextureIndexNormal.index], index: TextureIndexNormal.index)
                renderEncoder.setFragmentTexture(submesh.textures[TextureIndexSpecular.index], index: TextureIndexSpecular.index)

                let metalKitSubmesh = submesh.metalKitSubmesh

                renderEncoder.drawIndexedPrimitives(type: metalKitSubmesh.primitiveType,
                                                    indexCount: metalKitSubmesh.indexCount,
                                                    indexType: metalKitSubmesh.indexType,
                                                    indexBuffer: metalKitSubmesh.indexBuffer.buffer,
                                                    indexBufferOffset: metalKitSubmesh.indexBuffer.offset)
            }
        }
    }

    func draw(in view: MTKView) {
        // Wait to ensure only MaxBufferInFlight are getting processed by any stage in the
        // Metal pipeline (app, Metal, drivers, GPU, etc.).
        inFlightSemaphore.wait()

        // Create a new command buffer for each render pass to the current drawable.
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "MyCommand"

        // Add a completion handler that signals inFlightSemaphore when Metal and the GPU have
        // fully finished processing the commands encoded this frame. This callback indicates when
        // the dynamic buffers written in this frame are no longer needed by Metal and the GPU.
        commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
            inFlightSemaphore.signal()
        }

        self.updateFrameState()

        if let currentDrawable = view.currentDrawable {
            if NumSamples > 1 {
                viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].resolveTexture = currentDrawable.texture
            } /* else {
                viewRenderPassDescriptor.colorAttachments[RenderTargetLighting.index].texture = currentDrawable.texture
            }*/

            // Create a render command encoder to render to the drawable.
            guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: viewRenderPassDescriptor) else { return }

            renderEncoder.setCullMode(.back)

            // Render scene to depth buffer only. You later use this data to determine the minimum
            // and maximum depth values of each tile.
            renderEncoder.pushDebugGroup("Depth Pre-Pass")
            renderEncoder.setRenderPipelineState(depthPrePassPipelineState)
            renderEncoder.setDepthStencilState(defaultDepthState)
            renderEncoder.setVertexBuffer(frameDataBuffers[currentBufferIndex], offset: 0, index: BufferIndexFrameData.index)
            self.drawMeshes(renderEncoder: renderEncoder)

            // Calculate light bins.
            renderEncoder.pushDebugGroup("Calculate Light Bins")
            renderEncoder.setRenderPipelineState(lightBinCreationPipelineState)
            renderEncoder.setThreadgroupMemoryLength(Int(TileDataSize), offset: threadgroupBufferSize, index: ThreadgroupBufferIndexTileData.index)
            renderEncoder.dispatchThreadsPerTile(MTLSize(width: Int(TileWidth), height: Int(TileHeight), depth: 1))
            renderEncoder.popDebugGroup()

            // Perform tile culling, to minimize the number of lights rendered per tile.
            renderEncoder.pushDebugGroup("Prepare Light Lists")
            renderEncoder.setRenderPipelineState(lightCullingPipelineState)
            renderEncoder.setThreadgroupMemoryLength(threadgroupBufferSize, offset: 0, index: ThreadgroupBufferIndexLightList.index)
            renderEncoder.setThreadgroupMemoryLength(Int(TileDataSize), offset: threadgroupBufferSize, index: ThreadgroupBufferIndexTileData.index)
            renderEncoder.setTileBuffer(frameDataBuffers[currentBufferIndex], offset: 0, index: BufferIndexFrameData.index)
            renderEncoder.setTileBuffer(lightEyePositions[currentBufferIndex], offset: 0, index: BufferIndexLightsPosition.index)
            renderEncoder.dispatchThreadsPerTile(MTLSize(width: Int(TileWidth), height: Int(TileHeight), depth: 1))
            renderEncoder.popDebugGroup()

            // Render objects with lighting
            renderEncoder.pushDebugGroup("Render Forward Lighting")
            renderEncoder.setRenderPipelineState(forwardLightingPipelineState)
            renderEncoder.setDepthStencilState(relaxedDepthStencilState)
            renderEncoder.setVertexBuffer(frameDataBuffers[currentBufferIndex], offset: 0, index: BufferIndexFrameData.index)
            renderEncoder.setFragmentBuffer(frameDataBuffers[currentBufferIndex], offset: 0, index: BufferIndexFrameData.index)
            renderEncoder.setFragmentBuffer(lightsData, offset: 0, index: BufferIndexLightsData.index)
            renderEncoder.setFragmentBuffer(lightWorldPositions[currentBufferIndex], offset: 0, index: BufferIndexLightsPosition.index)
            self.drawMeshes(renderEncoder: renderEncoder)
            renderEncoder.popDebugGroup()

            // Draw fairies.
            renderEncoder.pushDebugGroup("Draw Fairies")
            renderEncoder.setRenderPipelineState(fairyPipelineState)
            renderEncoder.setDepthStencilState(defaultDepthState)
            renderEncoder.setVertexBuffer(frameDataBuffers[currentBufferIndex], offset: 0, index: BufferIndexFrameData.index)
            renderEncoder.setVertexBuffer(fairy, offset: 0, index: BufferIndexMeshPositions.index)
            renderEncoder.setVertexBuffer(lightsData, offset: 0, index: BufferIndexLightsData.index)
            renderEncoder.setVertexBuffer(lightWorldPositions[currentBufferIndex], offset: 0, index: BufferIndexLightsPosition.index)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: numFairyVertices, instanceCount: numLights)
            renderEncoder.popDebugGroup()

            renderEncoder.endEncoding()
        }

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }
}
