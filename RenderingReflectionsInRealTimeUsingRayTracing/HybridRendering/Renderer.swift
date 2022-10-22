import Foundation
import MetalKit
import MetalPerformanceShaders

private let maxFramesInFlight = 3
private let kMaxInstances = 4

private let kAlignedInstanceTransformsStructSize = (MemoryLayout<InstanceTransform>.stride & ~0xFF) + 0x100
private let kAlignedInstanceTransformStride = kAlignedInstanceTransformsStructSize / MemoryLayout<InstanceTransform>.stride

struct ModelInstance {
    let meshIndex: UInt32
    let position: SIMD3<Float>
    let rotationRad: Float
}

struct ThinGBuffer {
    let positionTexture: MTLTexture
    let directionTexture: MTLTexture
}

enum AccelerationStructureEvents: UInt64 {
    case primitiveAccelerationStructureBuild = 1
    case instanceAccelerationStructureBuild = 2
}

final class Renderer: NSObject {
    enum RenderMode {
        case noRaytracing
        case metalRaytracing
        case reflectionsOnly
    }

    private var renderMode: RenderMode = .noRaytracing

    private let inFlightSemaphore = DispatchSemaphore(value: maxFramesInFlight)

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let pipelineState: MTLRenderPipelineState
    private let pipelineStateNoRT: MTLRenderPipelineState
    private let pipelineStateReflOnly: MTLRenderPipelineState
    private let pipelineStateGBuffer: MTLRenderPipelineState
    private let pipelineStateSkybox: MTLRenderPipelineState

    private var rtReflectionMap: MTLTexture!
    private var rtReflectionFunction: MTLFunction! = nil
    private var rtReflectionPipeline: MTLComputePipelineState! = nil
    private let rtMipmapPipeline: MTLRenderPipelineState
    private let bloomThresholdPipeline: MTLRenderPipelineState
    private let postMergePipeline: MTLRenderPipelineState

    private let depthState: MTLDepthStencilState

    private let lightDataBuffer: MTLBuffer
    private let cameraDataBuffers: [MTLBuffer]
    private let instanceTransformBuffer: MTLBuffer

    private let accelerationStructureBuildEvent: MTLEvent
    private let modelInstances: [ModelInstance]

    private var instanceAccelerationStructure: MTLAccelerationStructure!

    private let mtlVertexDescriptor: MTLVertexDescriptor
    private let mtlSkyboxVertexDescriptor: MTLVertexDescriptor

    private var cameraBufferIndex: Int = 0
    private var projectionMatrix: simd_float4x4

    private var meshes: [MeshObj] = []
    private var skyMap: MTLTexture!
    private var skybox: MeshObj!

    private var sceneResources: [MTLResource] = []
    private var sceneArgumentBuffer: MTLBuffer!

    private var cameraAngle: Float = 0.0
    private var cameraPanSpeedFactor: Float = 0.5
    private var metallicBias: Float = 0.0
    private var roughnessBias: Float = 0.0
    private var exposure: Float = 1.0

    private var rawColorMap: MTLTexture!
    private var bloomThresholdMap: MTLTexture!
    private var bloomBlurMap: MTLTexture!

    private var thinGBuffer: ThinGBuffer!
    private var rtMipmappingHeap: MTLHeap!

    private var accelerationStructureHeap: MTLHeap!

    private var primitiveAccelerationStructures: [MTLAccelerationStructure]!

    static func initializeModelInstances() -> [ModelInstance] {
        assert(kMaxInstances == 4, "Expected 3 Model Instances")

        return [
            ModelInstance(meshIndex: 0, position: [20.0, -5.0, -40.0], rotationRad: 135 * Float.pi / 180.0),
            ModelInstance(meshIndex: 0, position: [-13.0, -5.0, -20.0], rotationRad: 235 * Float.pi / 180.0),
            ModelInstance(meshIndex: 1, position: [-5.0, 2.75, -55.0], rotationRad: 0.0),
            ModelInstance(meshIndex: 2, position: [0.0, -5.0, -0.0], rotationRad: 0.0)
        ]
    }

    static func projectionMatrix(aspect: Float) -> simd_float4x4 {
        simd_float4x4.perspectiveRightHand(fovyRadians: 65.0 * (Float.pi / 180.0),
                                           aspect: aspect, nearZ: 0.1, farZ: 250.0)
    }

    static func setVertexDescriptorAttribute(vertexDescriptor: inout MTLVertexDescriptor, attribute: VertexAttribute,
                                             format: MTLVertexFormat, offset: Int, bufferIndex: BufferIndex) {
        vertexDescriptor.attributes[Int(attribute.rawValue)].format = format
        vertexDescriptor.attributes[Int(attribute.rawValue)].offset = offset
        vertexDescriptor.attributes[Int(attribute.rawValue)].bufferIndex = Int(bufferIndex.rawValue)
    }

    static func createVertexDescriptor() -> MTLVertexDescriptor {
        var vertexDescriptor = MTLVertexDescriptor()

        // Positions.
        Self.setVertexDescriptorAttribute(vertexDescriptor: &vertexDescriptor,
                                          attribute: VertexAttributePosition,
                                          format: .float3, offset: 0, bufferIndex: BufferIndexMeshPositions)

        // Texture coordinates.
        Self.setVertexDescriptorAttribute(vertexDescriptor: &vertexDescriptor,
                                          attribute: VertexAttributeTexcoord,
                                          format: .float2, offset: 0, bufferIndex: BufferIndexMeshGenerics)

        // Normals.
        Self.setVertexDescriptorAttribute(vertexDescriptor: &vertexDescriptor,
                                          attribute: VertexAttributeNormal,
                                          format: .half4, offset: 8, bufferIndex: BufferIndexMeshGenerics)

        // Tangents.
        Self.setVertexDescriptorAttribute(vertexDescriptor: &vertexDescriptor,
                                          attribute: VertexAttributeTangent,
                                          format: .half4, offset: 16, bufferIndex: BufferIndexMeshGenerics)

        // Bitangents.
        Self.setVertexDescriptorAttribute(vertexDescriptor: &vertexDescriptor,
                                          attribute: VertexAttributeBitangent,
                                          format: .half4, offset: 24, bufferIndex: BufferIndexMeshGenerics)

        // Position Buffer Layout
        vertexDescriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stride = 12
        vertexDescriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stepRate = 1
        vertexDescriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stepFunction = .perVertex

        // Generic Attribute Buffer Layout
        vertexDescriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stride = 32
        vertexDescriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stepRate = 1
        vertexDescriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stepFunction = .perVertex

        return vertexDescriptor
    }

    static func createSkyboxVertexDescriptor() -> MTLVertexDescriptor {
        let descriptor = MTLVertexDescriptor()

        descriptor.attributes[Int(VertexAttributePosition.rawValue)].format = .float3
        descriptor.attributes[Int(VertexAttributePosition.rawValue)].offset = 0
        descriptor.attributes[Int(VertexAttributePosition.rawValue)].bufferIndex = Int(BufferIndexMeshPositions.rawValue)

        descriptor.attributes[Int(VertexAttributeTexcoord.rawValue)].format = .float2
        descriptor.attributes[Int(VertexAttributeTexcoord.rawValue)].offset = 0
        descriptor.attributes[Int(VertexAttributeTexcoord.rawValue)].bufferIndex = Int(BufferIndexMeshGenerics.rawValue)

        descriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stride = 12
        descriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stepRate = 1
        descriptor.layouts[Int(BufferIndexMeshPositions.rawValue)].stepFunction = .perVertex

        descriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stride = MemoryLayout<SIMD2<Float>>.stride
        descriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stepRate = 1
        descriptor.layouts[Int(BufferIndexMeshGenerics.rawValue)].stepFunction = .perVertex

        return descriptor
    }

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue
        self.accelerationStructureBuildEvent = device.makeEvent()!

        metalView.device = device

        metalView.clearColor = MTLClearColor(red: 0.3, green: 0.3, blue: 0.3, alpha: 1)
        metalView.depthStencilPixelFormat = .depth32Float_stencil8
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        metalView.preferredFramesPerSecond = 60

        self.modelInstances = Self.initializeModelInstances()
        self.projectionMatrix = Self.projectionMatrix(aspect: Float(metalView.bounds.width / metalView.bounds.height))

        // MARK: loadMetalWithView
        self.mtlVertexDescriptor = Self.createVertexDescriptor()
        self.mtlSkyboxVertexDescriptor = Self.createSkyboxVertexDescriptor()

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader") else { fatalError() }

        let functionConstants = MTLFunctionConstantValues()
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()

        // MARK: Make raytracing RPS
        var enableRaytracing = true

        functionConstants.setConstantValue(&enableRaytracing, type: .bool, index: Int(ConstantIndexRayTracingEnabled.rawValue))

        let fragmentFunction = try! defaultLibrary.makeFunction(name: "fragmentShader", constantValues: functionConstants)
        pipelineStateDescriptor.label = "RT Pipeline"
        pipelineStateDescriptor.rasterSampleCount = metalView.sampleCount
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .rg11b10Float
        pipelineStateDescriptor.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat
        pipelineStateDescriptor.stencilAttachmentPixelFormat = metalView.depthStencilPixelFormat

        self.pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        // MARK: Make non-RT RPS
        enableRaytracing = false
        functionConstants.setConstantValue(&enableRaytracing, type: .bool, index: Int(ConstantIndexRayTracingEnabled.rawValue))
        let fragmentFunctionNoRT = try! defaultLibrary.makeFunction(name: "fragmentShader", constantValues: functionConstants)
        pipelineStateDescriptor.label = "No RT Pipeline"
        pipelineStateDescriptor.fragmentFunction = fragmentFunctionNoRT
        pipelineStateNoRT = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        // MARK: Reflection shader RPS
        pipelineStateDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "reflectionShader")
        pipelineStateDescriptor.label = "Reflection Viewer Pipeline"
        pipelineStateReflOnly = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        // MARK: GBuffer RPS
        pipelineStateDescriptor.fragmentFunction = defaultLibrary.makeFunction(name: "gBufferFragmentShader")
        pipelineStateDescriptor.label = "Thin GBuffer Pipeline"
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        pipelineStateDescriptor.colorAttachments[1].pixelFormat = .rgba16Float
        pipelineStateGBuffer = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        // MARK: Skybox RPS
        let skyboxVertexFunction = defaultLibrary.makeFunction(name: "skyboxVertex")
        let skyboxFragmentFunction = defaultLibrary.makeFunction(name: "skyboxFragment")
        pipelineStateDescriptor.label = "Skybox Pipeline"
        pipelineStateDescriptor.vertexDescriptor = mtlSkyboxVertexDescriptor
        pipelineStateDescriptor.vertexFunction = skyboxVertexFunction
        pipelineStateDescriptor.fragmentFunction = skyboxFragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .rg11b10Float
        pipelineStateDescriptor.colorAttachments[1].pixelFormat = .invalid
        pipelineStateSkybox = try! device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        if device.supportsRaytracing {
            rtReflectionFunction = defaultLibrary.makeFunction(name: "rtReflection")
            rtReflectionPipeline = try! device.makeComputePipelineState(function: rtReflectionFunction)
            renderMode = .metalRaytracing
        } else {
            renderMode = .noRaytracing
        }

        // MARK: Passthrough
        let passthroughVert = defaultLibrary.makeFunction(name: "vertexPassthrough")
        var fragmentFn = defaultLibrary.makeFunction(name: "fragmentPassthrough")
        let passthroughDesc = MTLRenderPipelineDescriptor()
        passthroughDesc.label = "Passthrough Pipeline"
        passthroughDesc.vertexFunction = passthroughVert
        passthroughDesc.fragmentFunction = fragmentFn
        passthroughDesc.colorAttachments[0].pixelFormat = .rg11b10Float
        rtMipmapPipeline = try! device.makeRenderPipelineState(descriptor: passthroughDesc)

        fragmentFn = defaultLibrary.makeFunction(name: "fragmentBloomThreshold")
        passthroughDesc.fragmentFunction = fragmentFn
        passthroughDesc.colorAttachments[0].pixelFormat = .rg11b10Float
        passthroughDesc.label = "Fragment Bloom Threshold Pipeline"
        bloomThresholdPipeline = try! device.makeRenderPipelineState(descriptor: passthroughDesc)

        fragmentFn = defaultLibrary.makeFunction(name: "fragmentPostprocessMerge")
        passthroughDesc.label = "Fragment Postprocess Merge Pipeline"
        passthroughDesc.fragmentFunction = fragmentFn
        passthroughDesc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        postMergePipeline = try! device.makeRenderPipelineState(descriptor: passthroughDesc)

        let depthStateDesc = MTLDepthStencilDescriptor()
        depthStateDesc.depthCompareFunction = .less
        depthStateDesc.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: depthStateDesc)!

        var cameraDataBuffers: [MTLBuffer] = []

        for i in 0..<maxFramesInFlight {
            guard let buffer = device.makeBuffer(length: MemoryLayout<CameraData>.stride, options: .storageModeShared) else { fatalError() }
            buffer.label = "CameraDataBuffer \(i)"
            cameraDataBuffers.append(buffer)
        }

        self.cameraDataBuffers = cameraDataBuffers

        let instanceBufferSize = kAlignedInstanceTransformsStructSize * kMaxInstances
        instanceTransformBuffer = device.makeBuffer(length: instanceBufferSize, options: .storageModeShared)!
        instanceTransformBuffer.label = "Instance Transform Buffer"

        lightDataBuffer = device.makeBuffer(length: MemoryLayout<LightData>.stride, options: .storageModeShared)!

        super.init()

        setStaticState()
        loadAssets()

        buildSceneArgumentBufferMetal3()

        resizeRTReflectionMap(to: metalView.drawableSize)
        buildRTAccelerationStructures()

        metalView.delegate = self
    }

    private func buildRTAccelerationStructures() {
        // Each mesh is an individual primitive acceleration structure, with each submesh being one
        // geometry within that acceleration structure.

        // Instance Acceleration Structure references n instances.
        // 1 Instance references 1 Primitive Acceleration Structure
        // 1 Primitive Acceleration Structure = 1 Mesh in self.meshes
        // 1 Primitive Acceleration Structure -> n geometries == n submeshes
        var primitiveAccelerationDescriptors: [MTLPrimitiveAccelerationStructureDescriptor] = []

        for mesh in meshes {
            var geometries: [MTLAccelerationStructureTriangleGeometryDescriptor] = []
            for submesh in mesh.submeshes {
                let g = MTLAccelerationStructureTriangleGeometryDescriptor()
                g.vertexBuffer = mesh.metalKitMesh.vertexBuffers.first!.buffer
                g.vertexBufferOffset = mesh.metalKitMesh.vertexBuffers.first!.offset
                g.vertexStride = 12 // The buffer must be packed XYZ XYZ XYZ ...

                g.indexBuffer = submesh.metalKitSubmesh.indexBuffer.buffer
                g.indexBufferOffset = submesh.metalKitSubmesh.indexBuffer.offset
                g.indexType = submesh.metalKitSubmesh.indexType

                let indexElementSize = (g.indexType == .uint16) ? MemoryLayout<UInt16>.stride : MemoryLayout<UInt32>.stride
                g.triangleCount = submesh.metalKitSubmesh.indexBuffer.length / indexElementSize / 3
                geometries.append(g)
            }
            let primDesc = MTLPrimitiveAccelerationStructureDescriptor()
            primDesc.geometryDescriptors = geometries
            primitiveAccelerationDescriptors.append(primDesc)
        }

        // Allocate all primitive acceleration structures.
        // On Metal 3, allocate directly from a MTLHeap.
        let storageSizes = self.calculateSizeForPrimitiveAccelerationStructures(primitiveAccelerationDescriptors: primitiveAccelerationDescriptors)
        let heapDesc = MTLHeapDescriptor()
        heapDesc.size = storageSizes.accelerationStructureSize
        accelerationStructureHeap = device.makeHeap(descriptor: heapDesc)
        accelerationStructureHeap.label = "Acceleration Structure Heap"
        primitiveAccelerationStructures = allocateAndBuildAccelerationStructuresWithDescriptors(descriptors: primitiveAccelerationDescriptors, heap: accelerationStructureHeap, maxScratchBufferSize: storageSizes.buildScratchBufferSize, signalEvent: accelerationStructureBuildEvent)

        let instanceAccelStructureDesc = MTLInstanceAccelerationStructureDescriptor()
        instanceAccelStructureDesc.instancedAccelerationStructures = primitiveAccelerationStructures
        instanceAccelStructureDesc.instanceCount = kMaxInstances

        // Load instance data (two fire trucks, one sphere, and the floor)
        let instanceDescriptorBuffer = device.makeBuffer(length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * kMaxInstances)!
        let instanceDescriptors = instanceDescriptorBuffer.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: kMaxInstances)

        let pInstanceTransforms = instanceTransformBuffer.contents().bindMemory(to: InstanceTransform.self, capacity: kMaxInstances)

        for i in 0..<kMaxInstances {
            instanceDescriptors[i].accelerationStructureIndex = modelInstances[i].meshIndex
            instanceDescriptors[i].intersectionFunctionTableOffset = 0
            instanceDescriptors[i].mask = 0xFF
            instanceDescriptors[i].options = []

            let transforms = pInstanceTransforms[i * kAlignedInstanceTransformStride]
            instanceDescriptors[i].transformationMatrix = transforms.modelViewMatrix.dropLastRow()
        }

        instanceAccelStructureDesc.instanceDescriptorBuffer = instanceDescriptorBuffer

        let cmd = commandQueue.makeCommandBuffer()!
        cmd.encodeWaitForEvent(accelerationStructureBuildEvent, value: AccelerationStructureEvents.primitiveAccelerationStructureBuild.rawValue)

        instanceAccelerationStructure = allocateAndBuildAccelerationStructuresWithDescriptors(descriptor: instanceAccelStructureDesc, commandBuffer: cmd)
        cmd.encodeSignalEvent(accelerationStructureBuildEvent, value: AccelerationStructureEvents.instanceAccelerationStructureBuild.rawValue)
        cmd.commit()
    }

    private func allocateAndBuildAccelerationStructuresWithDescriptors(descriptor: MTLAccelerationStructureDescriptor,
                                                                       commandBuffer: MTLCommandBuffer) -> MTLAccelerationStructure {
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)!
        let accelStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!

        let enc = commandBuffer.makeAccelerationStructureCommandEncoder()!
        enc.build(accelerationStructure: accelStructure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()

        return accelStructure
    }

    private func allocateAndBuildAccelerationStructuresWithDescriptors(descriptors: [MTLAccelerationStructureDescriptor],
                                                                       heap: MTLHeap,
                                                                       maxScratchBufferSize: Int,
                                                                       signalEvent: MTLEvent) -> [MTLAccelerationStructure] {
        var accelStructures: [MTLAccelerationStructure] = []

        let scratch = device.makeBuffer(length: maxScratchBufferSize, options: .storageModePrivate)!
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeAccelerationStructureCommandEncoder()

        for descriptor in descriptors {
            let sizes = device.heapAccelerationStructureSizeAndAlign(descriptor: descriptor)
            let accelStructure = heap.makeAccelerationStructure(size: sizes.size)!
            enc?.build(accelerationStructure: accelStructure, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
            accelStructures.append(accelStructure)
        }

        enc?.endEncoding()
        cmd.encodeSignalEvent(signalEvent, value: AccelerationStructureEvents.primitiveAccelerationStructureBuild.rawValue)
        cmd.commit()

        return accelStructures
    }

    private func calculateSizeForPrimitiveAccelerationStructures(primitiveAccelerationDescriptors: [MTLPrimitiveAccelerationStructureDescriptor])
    -> MTLAccelerationStructureSizes {
        var totalSizes = MTLAccelerationStructureSizes(accelerationStructureSize: 0, buildScratchBufferSize: 0, refitScratchBufferSize: 0)

        for desc in primitiveAccelerationDescriptors {
            let sizeAndAlign = device.heapAccelerationStructureSizeAndAlign(descriptor: desc)
            let sizes = device.accelerationStructureSizes(descriptor: desc)
            totalSizes.accelerationStructureSize += (sizeAndAlign.size + sizeAndAlign.align)
            totalSizes.buildScratchBufferSize = max(sizes.buildScratchBufferSize, totalSizes.buildScratchBufferSize)
            totalSizes.refitScratchBufferSize = max(sizes.refitScratchBufferSize, totalSizes.refitScratchBufferSize)
        }

        return totalSizes
    }

    private func resizeRTReflectionMap(to size: CGSize) {
        let size = (size != .zero) ? size : CGSize(width: 1, height: 1)

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg11b10Float, width: Int(size.width), height: Int(size.height), mipmapped: true)
        desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
        rtReflectionMap = device.makeTexture(descriptor: desc)

        desc.mipmapLevelCount = 1
        rawColorMap = device.makeTexture(descriptor: desc)!
        bloomThresholdMap = device.makeTexture(descriptor: desc)!
        bloomBlurMap = device.makeTexture(descriptor: desc)!

        desc.pixelFormat = .rgba16Float
        desc.usage = [.shaderRead, .renderTarget]
        thinGBuffer = ThinGBuffer(positionTexture: device.makeTexture(descriptor: desc)!,
                                  directionTexture: device.makeTexture(descriptor: desc)!)

        let hd = MTLHeapDescriptor()
        hd.size = Int(size.width * size.height * 4 * 2 * 3)
        hd.storageMode = .private
        rtMipmappingHeap = device.makeHeap(descriptor: hd)
        rtMipmappingHeap.label = "RT Mipmapping Heap"
    }

    private func buildSceneArgumentBufferMetal3() {
        let storageMode: MTLResourceOptions = .storageModeShared

        // The renderer builds this structure to match the ray-traced scene structure so the
        // ray-tracing shader navigates it. In particular, Metal represents each submesh as a
        // geometry in the primitive acceleration structure.

        var sceneResources: [MTLResource] = []

        let instanceArgumentSize = MemoryLayout<Instance>.stride * kMaxInstances
        let instanceArgumentBuffer = newBuffer(label: "InstanceArgumentBuffer", length: instanceArgumentSize, options: storageMode)
        sceneResources.append(instanceArgumentBuffer)

        let instancePtr = instanceArgumentBuffer.contents().bindMemory(to: Instance.self, capacity: kMaxInstances)

        // Encode the instances array in 'Scene'
        for i in 0..<kMaxInstances {
            instancePtr[i].meshIndex = modelInstances[i].meshIndex
            instancePtr[i].transform = calculateTransform(instance: modelInstances[i])
        }

        let meshArgumentSize = MemoryLayout<Mesh>.stride * meshes.count
        let meshArgumentBuffer = newBuffer(label: "MeshArgumentBuffer", length: meshArgumentSize, options: storageMode)
        sceneResources.append(meshArgumentBuffer)

        // Encode the meshes array in Scene
        let meshPtr = meshArgumentBuffer.contents().bindMemory(to: Mesh.self, capacity: meshes.count)

        for i in 0..<meshes.count {
            let mesh = meshes[i]

            let metalKitMesh = mesh.metalKitMesh

            meshPtr[i].positions = metalKitMesh.vertexBuffers[0].buffer.gpuAddress + UInt64(metalKitMesh.vertexBuffers[0].offset)
            meshPtr[i].generics = metalKitMesh.vertexBuffers[1].buffer.gpuAddress + UInt64(metalKitMesh.vertexBuffers[1].offset)

            assert(metalKitMesh.vertexBuffers.count == 2, "Unsupported number of buffers")
            sceneResources.append(metalKitMesh.vertexBuffers[0].buffer)
            sceneResources.append(metalKitMesh.vertexBuffers[1].buffer)

            // Build submeshes into a buffer and reference it through a pointer in the mesh.
            let submeshArgumentSize = MemoryLayout<Submesh>.stride * mesh.submeshes.count
            let submeshArgumentBuffer = newBuffer(label: "SubmeshArgumentBuffer \(i)", length: submeshArgumentSize, options: storageMode)
            sceneResources.append(submeshArgumentBuffer)

            let pSubmesh = submeshArgumentBuffer.contents().bindMemory(to: Submesh.self, capacity: mesh.submeshes.count)

            for j in 0..<mesh.submeshes.count {
                let submesh = mesh.submeshes[j]

                let indexBuffer = submesh.metalKitSubmesh.indexBuffer
                pSubmesh[j].shortIndexType = submesh.metalKitSubmesh.indexType == .uint32 ? 0 : 1
                pSubmesh[j].indices = indexBuffer.buffer.gpuAddress + UInt64(indexBuffer.offset)

                for m in 0..<submesh.textures.count {
                    switch m {
                    case 0:
                        pSubmesh[j].materials.0 = submesh.textures[m].gpuResourceID
                    case 1:
                        pSubmesh[j].materials.1 = submesh.textures[m].gpuResourceID
                    case 2:
                        pSubmesh[j].materials.2 = submesh.textures[m].gpuResourceID
                    case 3:
                        pSubmesh[j].materials.3 = submesh.textures[m].gpuResourceID
                    case 4:
                        pSubmesh[j].materials.4 = submesh.textures[m].gpuResourceID
                    default: fatalError()
                    }
                }

                sceneResources.append(submesh.metalKitSubmesh.indexBuffer.buffer)
                sceneResources.append(contentsOf: submesh.textures)
            }

            meshPtr[i].submeshes = submeshArgumentBuffer.gpuAddress
        }

        let sceneArgumentBuffer = newBuffer(label: "SceneArgumentBuffer", length: MemoryLayout<Scene3D>.stride, options: storageMode)
        sceneResources.append(sceneArgumentBuffer)

        let pScene = sceneArgumentBuffer.contents().bindMemory(to: Scene3D.self, capacity: 1)

        pScene[0].instances = instanceArgumentBuffer.gpuAddress
        pScene[0].meshes = meshArgumentBuffer.gpuAddress

        self.sceneResources = sceneResources
        self.sceneArgumentBuffer = sceneArgumentBuffer
    }

    private func newBuffer(label: String, length: Int, options: MTLResourceOptions) -> MTLBuffer {
        let buffer = device.makeBuffer(length: length,options: options)!
        buffer.label = label
        return buffer
    }

    private func setStaticState() {
        var transforms = instanceTransformBuffer.contents().bindMemory(to: InstanceTransform.self, capacity: kMaxInstances)

        for i in 0..<kMaxInstances {
            transforms.pointee.modelViewMatrix = calculateTransform( instance: modelInstances[i] )
            transforms = transforms.advanced(by: kAlignedInstanceTransformStride)
        }

        updateCameraState()

        let lightData = lightDataBuffer.contents().bindMemory(to: LightData.self, capacity: 1)
        lightData.pointee.directionalLightInvDirection = -simd_normalize([0, -6, -6])
        lightData.pointee.lightIntensity = 5.0
    }

    private func calculateTransform(instance: ModelInstance) -> matrix_float4x4 {
        let rotationAxis: SIMD3<Float> = [0, 1, 0]
        let rotationMatrix = matrix_float4x4.rotation(instance.rotationRad, rotationAxis)
        let translationMatrix = matrix_float4x4.translation(instance.position)
        return translationMatrix * rotationMatrix
    }

    private func updateCameraState() {
        cameraBufferIndex = (cameraBufferIndex + 1) % maxFramesInFlight

        // Update Projection Matrix
        let cameraData = cameraDataBuffers[cameraBufferIndex].contents().bindMemory(to: CameraData.self, capacity: 1)
        cameraData.pointee.projectionMatrix = projectionMatrix

        // Update camera position (and view matrix)
        let camPos: SIMD3<Float> = [cosf(cameraAngle) * 10.0, 5, sinf(cameraAngle) * 22.5]
        cameraAngle += (0.02 * cameraPanSpeedFactor)
        if cameraAngle >= 2.0 * Float.pi {
            cameraAngle -= 2.0 * Float.pi
        }

        cameraData.pointee.viewMatrix = matrix_float4x4.translation(-camPos)
        cameraData.pointee.cameraPosition = camPos
        cameraData.pointee.metallicBias = metallicBias
        cameraData.pointee.roughnessBias = roughnessBias
    }

    func loadAssets() {
        // Create a Model I/O vertexDescriptor to format the Model I/O mesh vertices to
        // fit the Metal render pipeline's vertex descriptor layout.
        let modelIOVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        // Indicate the Metal vertex descriptor attribute mapping for each Model I/O attribute.

        (modelIOVertexDescriptor.attributes[Int(VertexAttributePosition.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributePosition
        (modelIOVertexDescriptor.attributes[Int(VertexAttributeTexcoord.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributeTextureCoordinate
        (modelIOVertexDescriptor.attributes[Int(VertexAttributeNormal.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributeNormal
        (modelIOVertexDescriptor.attributes[Int(VertexAttributeTangent.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributeTangent
        (modelIOVertexDescriptor.attributes[Int(VertexAttributeBitangent.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributeBitangent

        let modelFileURL = Bundle.main.url(forResource: "Models/firetruck.obj", withExtension: nil)!

        var scene: [MeshObj] = []

        scene.append(contentsOf: MeshObj.newMeshes(from: modelFileURL, modelIOVertexDescriptor: modelIOVertexDescriptor, metalDevice: device))
        scene.append(MeshObj(sphereWithRadius: 8.0, modelIOVertexDescriptor: modelIOVertexDescriptor, metalDevice: device))
        scene.append(MeshObj(planeWithDimensions: [200.0, 200.0], modelIOVertexDescriptor: modelIOVertexDescriptor, metalDevice: device))

        meshes = scene

        skyMap = TexturefromRadianceFile(filename: "kloppenheim_06_4k.hdr", device: device)

        let skyboxModelIOVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlSkyboxVertexDescriptor)
        (skyboxModelIOVertexDescriptor.attributes[Int(VertexAttributePosition.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributePosition
        (skyboxModelIOVertexDescriptor.attributes[Int(VertexAttributeTexcoord.rawValue)] as! MDLVertexAttribute).name = MDLVertexAttributeTextureCoordinate

        skybox = MeshObj(skyboxMeshOnDevice: device, vertexDescriptor: skyboxModelIOVertexDescriptor)
    }

    private func copyDepthStencilConfiguration(from src: MTLRenderPassDescriptor, to dest: inout MTLRenderPassDescriptor) {
        dest.depthAttachment.loadAction = src.depthAttachment.loadAction
        dest.depthAttachment.clearDepth = src.depthAttachment.clearDepth
        dest.depthAttachment.texture = src.depthAttachment.texture
        dest.stencilAttachment.loadAction = src.stencilAttachment.loadAction
        dest.stencilAttachment.clearStencil = src.stencilAttachment.clearStencil
        dest.stencilAttachment.texture = src.stencilAttachment.texture
    }

    private func encodeSceneRendering(renderEncoder: MTLRenderCommandEncoder) {
        // Flag the residency of indirect resources in the scene.
        for resource in sceneResources {
            renderEncoder.useResource(resource, usage: .read, stages: .fragment)
        }

        for i in 0..<kMaxInstances {
            let mesh = meshes[Int(modelInstances[i].meshIndex)]
            let metalKitMesh = mesh.metalKitMesh

            // Set the mesh's vertex buffers.
            for bufferIndex in 0..<metalKitMesh.vertexBuffers.count {
                let vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex]
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: bufferIndex)
            }

            // Draw each submesh of the mesh.
            for submeshIndex in 0..<mesh.submeshes.count {
                let submesh = mesh.submeshes[submeshIndex]

                // Access textures directly from the argument buffer and avoid rebinding them individually.
                // 'SubmeshKeypath' provides the path to the argument buffer containing the texture data
                // for this submesh. The shader navigates the scene argument buffer using this key
                // to find the textures.
                var submeshKeypath = SubmeshKeypath(instanceID: UInt32(i), submeshID: UInt32(submeshIndex))

                let metalKitSubmesh = submesh.metalKitSubmesh

                renderEncoder.setVertexBuffer(instanceTransformBuffer, offset: kAlignedInstanceTransformsStructSize * i, index: Int(BufferIndexInstanceTransforms.rawValue))

                renderEncoder.setVertexBuffer(cameraDataBuffers[cameraBufferIndex], offset: 0, index: Int(BufferIndexCameraData.rawValue))
                renderEncoder.setFragmentBuffer(cameraDataBuffers[cameraBufferIndex], offset: 0, index: Int(BufferIndexCameraData.rawValue))
                renderEncoder.setFragmentBuffer(lightDataBuffer, offset: 0, index: Int(BufferIndexLightData.rawValue))

                // Bind the scene and provide the keypath to retrieve this submesh's data.
                renderEncoder.setFragmentBuffer(sceneArgumentBuffer, offset: 0, index: Int(SceneIndex.rawValue))
                renderEncoder.setFragmentBytes(&submeshKeypath, length: MemoryLayout<SubmeshKeypath>.stride, index: Int(BufferIndexSubmeshKeypath.rawValue))

                renderEncoder.drawIndexedPrimitives(type: metalKitSubmesh.primitiveType,
                                                    indexCount: metalKitSubmesh.indexCount,
                                                    indexType: metalKitSubmesh.indexType,
                                                    indexBuffer: metalKitSubmesh.indexBuffer.buffer,
                                                    indexBufferOffset: metalKitSubmesh.indexBuffer.offset)
            }
        }
    }

    private func generateGaussMipmapsForTexture(texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let gauss = MPSImageGaussianBlur(device: device, sigma: 5.0)
        let tmpDesc = MTLTextureDescriptor()
        tmpDesc.textureType = .type2D
        tmpDesc.pixelFormat = .rg11b10Float
        tmpDesc.mipmapLevelCount = 1
        tmpDesc.usage = [.shaderRead, .shaderWrite]
        tmpDesc.resourceOptions = .storageModePrivate

        var src = rtReflectionMap!

        var newW = rtReflectionMap.width
        var newH = rtReflectionMap.height

        let event = device.makeEvent()!
        var count: UInt64 = 0
        commandBuffer.encodeSignalEvent(event, value: count)

        while count + 1 < rtReflectionMap.mipmapLevelCount {
            commandBuffer.pushDebugGroup("Mip level: \(count)")

            tmpDesc.width = newW
            tmpDesc.height = newH

            let dst = rtMipmappingHeap.makeTexture(descriptor: tmpDesc)!

            gauss.encode(commandBuffer: commandBuffer, sourceTexture: src, destinationTexture: dst)
            count += 1
            commandBuffer.encodeSignalEvent(event, value: count)
            commandBuffer.encodeWaitForEvent(event, value: count)

            let targetMip = rtReflectionMap.__newTextureView(with: .rg11b10Float, textureType: .type2D, levels: NSMakeRange(Int(count), 1), slices: NSMakeRange(0, 1))!

            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].loadAction = .dontCare
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[0].texture = targetMip

            let blit = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
            blit.setCullMode(.none)
            blit.setRenderPipelineState(rtMipmapPipeline)
            blit.setFragmentTexture(dst, index: 0)
            blit.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            blit.endEncoding()

            src = targetMip

            newW = newW / 2
            newH = newH / 2

            commandBuffer.popDebugGroup()
        }
    }

    private func projectionMatrix(aspect: Float) -> matrix_float4x4 {
        return simd_float4x4.perspectiveRightHand(fovyRadians: 65.0 * (.pi / 180.0), aspect: aspect, nearZ: 0.1, farZ: 250.0)
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        let aspect = size.width / size.height
        projectionMatrix = projectionMatrix(aspect: Float(aspect))

        self.resizeRTReflectionMap(to: size)
    }

    func draw(in view: MTKView) {
        inFlightSemaphore.wait()

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Render Commands"
        commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
            inFlightSemaphore.signal()
        }

        updateCameraState()

        // Delay getting the currentRenderPassDescriptor until the renderer absolutely needs it
        // to avoid holding on to the drawable and blocking the display pipeline any longer
        // than necessary.
        if let renderPassDescriptor = view.currentRenderPassDescriptor {

            // When ray tracing is in an enabled state, first render a thin G-Buffer
            // that contains position and reflection direction data. Then, dispatch a
            // compute kernel that ray traces mirror-like reflections from this data.
            if (renderMode == .metalRaytracing || renderMode == .reflectionsOnly) {
                var gBufferPass = MTLRenderPassDescriptor()
                gBufferPass.colorAttachments[0].loadAction = .clear
                gBufferPass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
                gBufferPass.colorAttachments[0].storeAction = .store
                gBufferPass.colorAttachments[0].texture = thinGBuffer.positionTexture

                gBufferPass.colorAttachments[1].loadAction = .clear
                gBufferPass.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
                gBufferPass.colorAttachments[1].storeAction = .store
                gBufferPass.colorAttachments[1].texture = thinGBuffer.directionTexture

                copyDepthStencilConfiguration(from: renderPassDescriptor, to: &gBufferPass)
                gBufferPass.depthAttachment.storeAction = .store

                // Create a render command encoder.
                commandBuffer.pushDebugGroup("Render Thin G-Buffer")
                let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: gBufferPass)!

                renderEncoder.label = "ThinGBufferRenderEncoder"

                // Set the render command encoder state.
                renderEncoder.setCullMode(.front)
                renderEncoder.setFrontFacing(.clockwise)
                renderEncoder.setRenderPipelineState(pipelineStateGBuffer)
                renderEncoder.setDepthStencilState(depthState)

                // Encode all draw calls for the scene.
                encodeSceneRendering(renderEncoder: renderEncoder)

                // Finish encoding commands.
                renderEncoder.endEncoding()
                commandBuffer.popDebugGroup()

                // The ray-traced reflections.
                commandBuffer.pushDebugGroup("Raytrace Compute")
                commandBuffer.encodeWaitForEvent(accelerationStructureBuildEvent, value: AccelerationStructureEvents.instanceAccelerationStructureBuild.rawValue)
                let compEnc = commandBuffer.makeComputeCommandEncoder()!
                compEnc.label = "RaytracedReflectionsComputeEncoder"
                compEnc.setTexture(rtReflectionMap, index: Int(OutImageIndex.rawValue))
                compEnc.setTexture(thinGBuffer.positionTexture, index: Int(ThinGBufferPositionIndex.rawValue))
                compEnc.setTexture(thinGBuffer.directionTexture, index: Int(ThinGBufferDirectionIndex.rawValue))
                compEnc.setTexture(skyMap, index: Int(SkyDomeTexture.rawValue))

                // Bind the root of the argument buffer for the scene.
                compEnc.setBuffer(sceneArgumentBuffer, offset: 0, index: Int(SceneIndex.rawValue))

                // Bind the prebuilt acceleration structure.
                compEnc.setAccelerationStructure(instanceAccelerationStructure, bufferIndex: Int(AccelerationStructureIndex.rawValue))

                compEnc.setBuffer(instanceTransformBuffer, offset: 0, index: Int(BufferIndexInstanceTransforms.rawValue))
                compEnc.setBuffer(cameraDataBuffers[cameraBufferIndex], offset: 0, index: Int(BufferIndexCameraData.rawValue))
                compEnc.setBuffer(lightDataBuffer, offset: 0, index: Int(BufferIndexLightData.rawValue))

                // Set the ray tracing reflection kernel.
                compEnc.setComputePipelineState(rtReflectionPipeline)

                // Flag residency for indirectly referenced resources.
                // These are:
                // 1. All primitive acceleration structures.
                // 2. Buffers and textures referenced through argument buffers.

                if let accelerationStructureHeap {
                    // Heap backs the acceleration structures. Mark the entire heap resident.
                    compEnc.useHeap(accelerationStructureHeap)
                } else {
                    // Acceleration structures are independent. Mark each one resident.
                    for primAccelStructure in primitiveAccelerationStructures {
                        compEnc.useResource(primAccelStructure, usage: .read)
                    }
                }

                for resource in sceneResources {
                    compEnc.useResource(resource, usage: .read)
                }

                // Determine the dispatch grid size and dispatch compute.
                let w = rtReflectionPipeline.threadExecutionWidth
                let h = rtReflectionPipeline.maxTotalThreadsPerThreadgroup / w
                let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
                let threadsPerGrid = MTLSize(width: rtReflectionMap.width, height: rtReflectionMap.height, depth: 1)

                compEnc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

                compEnc.endEncoding()
                commandBuffer.popDebugGroup()

                // Generally, for accurate rough reflections, a renderer performs cone ray tracing in
                // the ray tracing kernel. In this case, the renderer simplifies this by blurring the
                // mirror-like reflections along the mipchain. The renderer later biases the miplevel
                // that the GPU samples when reading the reflection in the accumulation pass.
                commandBuffer.pushDebugGroup("Generate Reflection Mipmaps")
                generateGaussMipmapsForTexture(texture: rtReflectionMap, commandBuffer: commandBuffer)
                commandBuffer.popDebugGroup()
            }

            // Encode the forward pass.
            let rpd = view.currentRenderPassDescriptor!
            let drawableTexture = rpd.colorAttachments[0].texture
            rpd.colorAttachments[0].texture = rawColorMap

            commandBuffer.pushDebugGroup("Forward Scene Render")
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
            renderEncoder.label = "ForwardPassRenderEncoder"

            if renderMode == .metalRaytracing {
                renderEncoder.setRenderPipelineState(pipelineState)
            } else if renderMode == .noRaytracing {
                renderEncoder.setRenderPipelineState(pipelineStateNoRT)
            } else if renderMode == .reflectionsOnly {
                renderEncoder.setRenderPipelineState(pipelineStateReflOnly)
            }

            renderEncoder.setCullMode(.front)
            renderEncoder.setFrontFacing(.clockwise)
            renderEncoder.setDepthStencilState(depthState)
            renderEncoder.setFragmentTexture(rtReflectionMap, index: Int(TextureIndexReflections.rawValue))
            renderEncoder.setFragmentTexture(skyMap, index: Int(SkyDomeTexture.rawValue))

            self.encodeSceneRendering(renderEncoder: renderEncoder)

            // Encode the skybox rendering
            renderEncoder.setCullMode(.back)
            renderEncoder.setRenderPipelineState(pipelineStateSkybox)

            renderEncoder.setVertexBuffer(cameraDataBuffers[cameraBufferIndex], offset: 0, index: Int(BufferIndexCameraData.rawValue))

            renderEncoder.setFragmentTexture(skyMap, index: 0)

            let metalKitMesh = skybox.metalKitMesh
            for bufferIndex in 0..<metalKitMesh.vertexBuffers.count {
                let vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex]
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: bufferIndex)
            }

            for submesh in metalKitMesh.submeshes {
                renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                    indexCount: submesh.indexCount,
                                                    indexType: submesh.indexType,
                                                    indexBuffer: submesh.indexBuffer.buffer,
                                                    indexBufferOffset: submesh.indexBuffer.offset)
            }

            renderEncoder.endEncoding()
            commandBuffer.popDebugGroup()

            // Clamp values to the bloom threshold
            func clampBloom() {
                commandBuffer.pushDebugGroup("Bloom Threshold")
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].loadAction = .dontCare
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].texture = bloomThresholdMap

                let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
                renderEncoder.pushDebugGroup("Postprocessing")
                renderEncoder.setRenderPipelineState(bloomThresholdPipeline)
                renderEncoder.setFragmentTexture(rawColorMap, index: 0)

                var threshold: Float = 2.0
                renderEncoder.setFragmentBytes(&threshold, length: MemoryLayout<Float>.stride, index: 0)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                renderEncoder.popDebugGroup()
                renderEncoder.endEncoding()
                commandBuffer.popDebugGroup()
            }

            clampBloom()

            func blurBloom() {
                commandBuffer.pushDebugGroup("Bloom Blur")
                let blur = MPSImageGaussianBlur(device: device, sigma: 5.0)
                blur.encode(commandBuffer: commandBuffer, sourceTexture: bloomThresholdMap, destinationTexture: bloomBlurMap)
            }

            blurBloom()

            func mergePostProcessing() {
                commandBuffer.pushDebugGroup("Final Merge")
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].loadAction = .dontCare
                rpd.colorAttachments[0].storeAction = .store
                rpd.colorAttachments[0].texture = drawableTexture

                let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
                renderEncoder.pushDebugGroup("Postprocessing Merge")
                renderEncoder.setRenderPipelineState(postMergePipeline)
                renderEncoder.setFragmentBytes(&exposure, length: MemoryLayout<Float>.stride, index: 0)
                renderEncoder.setFragmentTexture(rawColorMap, index: 0)
                renderEncoder.setFragmentTexture(bloomBlurMap, index: 1)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

                renderEncoder.popDebugGroup()
                renderEncoder.endEncoding()
                commandBuffer.popDebugGroup()
            }

            mergePostProcessing()

            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
        }

        commandBuffer.commit()
    }
}
