import MetalKit

let kNumObjectsX = Int(AAPLNumObjectsX)
let kNumObjectsY = Int(AAPLNumObjectsY)
let kNumObjectsZ = Int(AAPLNumObjectsZ)
let kNumObjects = Int(AAPLNumObjects)
let kLODCount = 3
let kMaxMeshletVertexCount = Int(AAPLMaxMeshletVertexCount)
let kMaxPrimitiveCount = Int(AAPLMaxPrimitiveCount)

let kMaxTotalThreadsPerObjectThreadgroup = 1
let kMaxTotalThreadsPerMeshThreadgroup = kMaxPrimitiveCount
let kMaxThreadgroupsPerMeshGrid = 8

typealias AAPLIndexType = UInt16

struct AAPLMeshInfo {
    let numLODs: UInt16 = 3
    let patchIndex: UInt16
    let color: simd_float4

    let vertexCount: UInt16

    let lod1: AAPLIndexRange
    let lod2: AAPLIndexRange
    let lod3: AAPLIndexRange
}

func matrix4x4_translation(tx: Float, ty: Float, tz: Float) -> matrix_float4x4 {
    return simd_matrix_from_rows(simd_make_float4(1, 0, 0, tx),
                                 simd_make_float4(0, 1, 0, ty),
                                 simd_make_float4(0, 0, 1, tz),
                                 simd_make_float4(0, 0, 0, 1))
}

func matrix_perspective_right_hand(fovyRadians: Float, aspect: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {

    let ys = 1.0 / tanf(fovyRadians * 0.5)
    let xs = ys / aspect
    let zs = farZ / (nearZ - farZ)
    return simd_matrix_from_rows(simd_make_float4(xs, 0, 0, 0),
                                 simd_make_float4(0, ys, 0, 0),
                                 simd_make_float4(0, 0, zs, nearZ * zs),
                                 simd_make_float4(0, 0, -1, 0))
}

func matrix4x4_YRotate(angleRadians: Float) -> matrix_float4x4 {
    let a = angleRadians
    return simd_matrix_from_rows(simd_make_float4(cosf(a), 0, sin(a), 0),
                                 simd_make_float4(0, 1, 0, 0),
                                 simd_make_float4(-sinf(a), 0, cosf(a), 0),
                                 simd_make_float4(0, 0, 0, 1))
}

final class Renderer: NSObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    let useMultisampleAntialiasing = true

    // Buffers
    private let transformsBuffer: MTLBuffer
    private let meshColorsBuffer: MTLBuffer
    private let meshVerticesBuffer: MTLBuffer
    private let meshIndicesBuffer: MTLBuffer
    private let meshInfoBuffer: MTLBuffer

    private var renderPipelineState: [MTLRenderPipelineState] = []
    private var depthStencilState: MTLDepthStencilState?

    private var meshVertices: [AAPLVertex] = []
    private var meshIndices: [AAPLIndexType] = []
    private var meshInfo: [AAPLMeshInfo] = []

    var offsetY: Float = 0
    var offsetZ: Float = 0
    var projectionMatrix: matrix_float4x4!
    var rotationSpeed: Float = 0.25
    var topologyChoice = 2
    var lodChoice = 0
    private var degree: Float = 0.0

    init(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            fatalError()
        }

        metalView.device = device
        metalView.depthStencilPixelFormat = .depth32Float_stencil8
        metalView.sampleCount = 4

        self.device = device
        self.commandQueue = commandQueue

        self.transformsBuffer = device.makeBuffer(length: kNumObjects * MemoryLayout<matrix_float4x4>.stride, options: .storageModeManaged)!
        self.meshColorsBuffer = device.makeBuffer(length: kNumObjects * MemoryLayout<vector_float3>.stride, options: .storageModeManaged)!

        self.meshVerticesBuffer = device.makeBuffer(length: kNumObjects * MemoryLayout<AAPLVertex>.stride * kMaxMeshletVertexCount * kLODCount, options: .storageModeManaged)!
        self.meshIndicesBuffer = device.makeBuffer(length: kNumObjects * MemoryLayout<AAPLIndexType>.stride * kMaxPrimitiveCount * 6 * kLODCount, options: .storageModeManaged)!
        self.meshInfoBuffer = device.makeBuffer(length: kNumObjects * MemoryLayout<AAPLMeshInfo>.stride, options: .storageModeManaged)!

        super.init()

        metalView.delegate = self
        metalView.clearColor = MTLClearColor(red: 0.65, green: 0.75, blue: 0.85, alpha: 1.0)
        metalView.preferredFramesPerSecond = 120

        buildShaders()
        makeMeshlets()
    }

    private func bernsteinBasisCubic(_ u: Float, _ i: Int) -> Float {
        let nChooseI: [Float] = [1.0, 3.0, 3.0, 1.0]
        return nChooseI[i] * powf(u, Float(i)) * powf(1.0 - u, Float(3 - i))
    }

    private func bicubicPoint(_ u: Float, _ v: Float, controlPoints: [simd_float3]) -> simd_float4 {
        var p = simd_make_float3(0, 0, 0)
        var k = 0
        for i in 0...3 {
            for j in 0...3 {
                let bNI_U = bernsteinBasisCubic(u, i)
                let bMJ_V = bernsteinBasisCubic(v, j)
                p += bNI_U * bMJ_V * controlPoints[k]
                k += 1
            }
        }
        return simd_make_float4(p.x, p.y, p.z, 1.0)
    }

    private var gshape = -1
    private var controlPoints: [simd_float3] = []

    private func bicubicPatch(shape: Int, u: Float, v: Float) -> simd_float4 {
        if (gshape != shape) {
            controlPoints = []
            // Generate control points if this object wasn't used before.
            for i in 0..<4 {
                for j in 0..<4 {
                    controlPoints.append(simd_float3(x: Float(i) / 3.0 - 0.5,
                                                     y: Float(j) / 3.0 - 0.5,
                                                     z: Float.random(in: -0.5...0.0)))
                }
            }
            gshape = shape
        }
        return bicubicPoint(u, v, controlPoints: controlPoints)
    }

    private func bicubicPatch3(shape: Int, u: Float, v: Float) -> simd_float3 {
        let p = bicubicPatch(shape: shape, u: u, v: v)
        return simd_make_float3(p)
    }

    private func buildShaders() {
        guard let library = device.makeDefaultLibrary() else { fatalError() }

        let meshDesc = MTLMeshRenderPipelineDescriptor()
        let fragmentFunction = library.makeFunction(name: "fragmentShader")!

        meshDesc.fragmentFunction = fragmentFunction
        meshDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        meshDesc.depthAttachmentPixelFormat = .depth16Unorm

        if useMultisampleAntialiasing {
            meshDesc.rasterSampleCount = 4
        }

        meshDesc.maxTotalThreadgroupsPerMeshGrid = kMaxThreadgroupsPerMeshGrid
        meshDesc.maxTotalThreadsPerObjectThreadgroup = kMaxTotalThreadsPerObjectThreadgroup
        meshDesc.maxTotalThreadsPerMeshThreadgroup = kMaxTotalThreadsPerMeshThreadgroup

        let meshShaders = ["meshShaderMeshStageFunctionPoints", "meshShaderMeshStageFunctionLines", "meshShaderMeshStageFunction"]

        for i in 0..<3 {
            var topology = i
            let constantValues = MTLFunctionConstantValues()
            constantValues.setConstantValue(&topology, type: .int, index: Int(AAPL_FUNCTION_CONSTANT_TOPOLOGY))

            let meshFunction = try! library.makeFunction(name: meshShaders[i], constantValues: constantValues)
            meshDesc.meshFunction = meshFunction
            let objectFunction = try! library.makeFunction(name: "meshShaderObjectStageFunction", constantValues: constantValues)
            meshDesc.objectFunction = objectFunction

            let (newState, _) = try! device.makeRenderPipelineState(descriptor: meshDesc, options: .argumentInfo)
            renderPipelineState.append(newState)
        }

        let depthStencilDesc = MTLDepthStencilDescriptor()
        depthStencilDesc.depthCompareFunction = .less
        depthStencilDesc.isDepthWriteEnabled = true
        self.depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDesc)
    }

    private func makePatchVertices(shape: Int, segmentsX: UInt32, segmentsY: UInt32, vertices: inout [AAPLVertex]) -> UInt32 {
        let vertexCount = segmentsX * segmentsY

        for j in 0..<segmentsY {
            for i in 0..<segmentsX {
                let u = Float(i) / Float(segmentsX - 1)
                let v = Float(j) / Float(segmentsY - 1)
                let position = bicubicPatch(shape: shape, u: u, v: v)
                let u1 = bicubicPatch3(shape: shape, u: u - 0.01, v: v)
                let u2 = bicubicPatch3(shape: shape, u: u + 0.01, v: v)
                let v1 = bicubicPatch3(shape: shape, u: u, v: v - 0.01)
                let v2 = bicubicPatch3(shape: shape, u: u, v: v + 0.01)
                let du = u2 - u1
                let dv = v2 - v1
                let N = simd_normalize(simd_cross(du, dv))
                let normal = simd_make_float4(N.x, N.y, N.z, 0)
                let uv = simd_make_float2(Float(i) / Float(segmentsX), Float(j) / Float(segmentsY))

                vertices.append(AAPLVertex(position: position, normal: normal, uv: uv))
            }
        }

        return vertexCount
    }

    private func makePatchIndices(segmentsX: UInt32, segmentsY: UInt32, startIndex: Int, indices: inout [AAPLIndexType]) -> Int {
        let indexCount = (segmentsX - 1) * (segmentsY - 1) * 6

        for j in 0..<(segmentsY - 1) {
            for i in 0..<(segmentsX - 1) {
                indices.append(UInt16(((j + 0) * segmentsX) + ((i + 1))))
                indices.append(UInt16(((j + 1) * segmentsX) + ((i + 0))))
                indices.append(UInt16(((j + 0) * segmentsX) + ((i + 0))))

                indices.append(UInt16(((j + 1) * segmentsX) + ((i + 1))))
                indices.append(UInt16(((j + 1) * segmentsX) + ((i + 0))))
                indices.append(UInt16(((j + 0) * segmentsX) + ((i + 1))))
            }
        }

        return Int(indexCount)
    }

    private func addLODs(startVertexIndex: Int, segX: UInt32, segY: UInt32, meshIndices: inout [AAPLIndexType]) -> AAPLIndexRange {
        let startIndex = meshIndices.count
        let lastIndex = startIndex + makePatchIndices(segmentsX: segX, segmentsY: segY, startIndex: startIndex, indices: &meshIndices)
        let vertexCount = meshIndices[startIndex..<lastIndex].max()! + 1

        return AAPLIndexRange(startIndex: UInt32(startIndex),
                              lastIndex: UInt32(lastIndex),
                              startVertexIndex: UInt32(startVertexIndex),
                              vertexCount: UInt32(vertexCount),
                              primitiveCount: UInt32((lastIndex - startIndex) / 3))
    }

    private func makeMeshlets() {
        let segX = AAPLNumPatchSegmentsX
        let segY = AAPLNumPatchSegmentsY
        meshVertices = []
        meshIndices = []
        meshInfo = []

        for i in 0..<AAPLNumObjects {
            let color = simd_make_float4(1.0, 0.0, 1.0, 1.0)

            // Add first LOD
            var startVertexIndices = meshVertices.count
            var vertexCount = makePatchVertices(shape: Int(i), segmentsX: segX, segmentsY: segY, vertices: &meshVertices)
            let lod1 = addLODs(startVertexIndex: startVertexIndices, segX: segX, segY: segY, meshIndices: &meshIndices)

            // Add second LOD
            startVertexIndices = meshVertices.count
            vertexCount += makePatchVertices(shape: Int(i), segmentsX: 5, segmentsY: 5, vertices: &meshVertices)
            let lod2 = addLODs(startVertexIndex: startVertexIndices, segX: 5, segY: 5, meshIndices: &meshIndices)

            // Add third LOD
            startVertexIndices = meshVertices.count
            vertexCount += makePatchVertices(shape: Int(i), segmentsX: 3, segmentsY: 3, vertices: &meshVertices)
            let lod3 = addLODs(startVertexIndex: startVertexIndices, segX: 3, segY: 3, meshIndices: &meshIndices)

            meshInfo.append(AAPLMeshInfo(patchIndex: UInt16(i), color: color, vertexCount: UInt16(vertexCount), lod1: lod1, lod2: lod2, lod3: lod3))
        }

        assert(meshVerticesBuffer.length >= meshVertices.count * MemoryLayout<AAPLVertex>.stride)
        assert(meshIndicesBuffer.length >= meshIndices.count * MemoryLayout<AAPLIndexType>.stride)

        memcpy(meshVerticesBuffer.contents(), &meshVertices, MemoryLayout<AAPLVertex>.stride * meshVertices.count)
        memcpy(meshIndicesBuffer.contents(), &meshIndices, MemoryLayout<AAPLIndexType>.stride * meshIndices.count)
        memcpy(meshInfoBuffer.contents(), &meshInfo, MemoryLayout<AAPLMeshInfo>.stride * meshInfo.count)

        meshVerticesBuffer.didModifyRange(0..<meshVerticesBuffer.length)
        meshIndicesBuffer.didModifyRange(0..<meshIndicesBuffer.length)
        meshInfoBuffer.didModifyRange(0..<meshInfoBuffer.length)
    }

    private func updateStage() {
        let transforms = transformsBuffer.contents().bindMemory(to: matrix_float4x4.self, capacity: kNumObjects)
        let meshColors = meshColorsBuffer.contents().bindMemory(to: simd_float3.self, capacity: kNumObjects)

        degree += rotationSpeed * Float.pi / 180.0

        let xDiv: Float = 1.0 / Float(kNumObjectsX + 1)
        let yDiv: Float = 1.0 / Float(kNumObjectsY)
        let zDiv: Float = 1.0 / Float(kNumObjectsZ)

        var count = 0
        for x in 0..<kNumObjectsX {
            let xPos = 2.0 * (Float(x) - (Float(kNumObjectsX - 1) / 2))

            for y in 0..<kNumObjectsY {
                let yPos = 2.0 * (Float(y) - (Float(kNumObjectsY - 1) / 2))

                for z in 0..<kNumObjectsZ {
                    let zPos = -12.0 - Float(z) * 2.0
                    transforms[count] = matrix_multiply(matrix4x4_translation(tx: xPos, ty: yPos, tz: zPos),
                                                        matrix4x4_YRotate(angleRadians: degree))
                    meshColors[count] = simd_make_float3((Float(x) + 1.0) * xDiv,
                                                         Float(y) * yDiv,
                                                         (1.0 + Float(z)) * zDiv)
                    meshColors[count] = simd_normalize(meshColors[count]) * 0.75
                    count += 1
                }
            }
        }

        transformsBuffer.didModifyRange(0..<transformsBuffer.length)
        meshColorsBuffer.didModifyRange(0..<meshColorsBuffer.length)
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: 65.0 * (Float.pi / 180.0), aspect: aspect, nearZ: 0.1, farZ: 100.0)
    }

    func draw(in view: MTKView) {
        // Get a command buffer and start a render command encoder.
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPassDesc = view.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc)
        else { fatalError() }

        let viewMatrix = matrix4x4_translation(tx: 0, ty: offsetY, tz: -10 + 10 * offsetZ)
        var viewProjectionMatrix = matrix_multiply(projectionMatrix, viewMatrix)

        updateStage()

        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setRenderPipelineState(renderPipelineState[topologyChoice])
        renderEncoder.setDepthStencilState(depthStencilState)

        renderEncoder.setObjectBuffer(meshVerticesBuffer, offset: 0, index: Int(AAPLBufferIndexMeshVertices.rawValue))
        renderEncoder.setObjectBuffer(meshIndicesBuffer, offset: 0, index: Int(AAPLBufferIndexMeshIndices.rawValue))
        renderEncoder.setObjectBuffer(meshInfoBuffer, offset: 0, index: Int(AAPLBufferIndexMeshInfo.rawValue))

        renderEncoder.setObjectBuffer(transformsBuffer, offset: 0, index: Int(AAPLBufferIndexTransforms.rawValue))
        renderEncoder.setObjectBuffer(meshColorsBuffer, offset: 0, index: Int(AAPLBufferIndexMeshColor.rawValue))
        renderEncoder.setObjectBytes(&viewProjectionMatrix, length: MemoryLayout<matrix_float4x4>.size, index: Int(AAPLBufferViewProjectionMatrix.rawValue))
        renderEncoder.setObjectBytes(&lodChoice, length: 4, index: Int(AAPLBufferIndexLODChoice.rawValue))

        renderEncoder.setMeshBytes(&viewProjectionMatrix, length: MemoryLayout<matrix_float4x4>.size, index: Int(AAPLBufferViewProjectionMatrix.rawValue))

        /// Draw objects using the mesh shaders.
        /// Parameter 1: threadgroupsPerGrid ... X=`AAPLNumObjectsX`, Y=`AAPLNumObjectsY`, ...
        /// Parameter 2: threadsPerObjectThreadgroup ... `AAPLMaxTotalThreadsPerObjectThreadgroup`
        /// Parameter 3: threadsPerMeshGroup ... X = `AAPLMaxTotalThreadsPerMeshThreadgroup` has a limit of 64 vertices per meshlet in this sample.
        ///
        /// The object shader copies vertices, indices, and other relevant data to the payload and generates the submesh groups.
        /// The parameter `positionInGrid` (`threadgroup_position_in_grid`) in the shader addresses the submesh.
        /// This tells the object shader the index of the transform for the submesh.
        /// The mesh shader uses the payload to generate the primitives (points, lines, or triangles).
        renderEncoder.drawMeshThreadgroups(MTLSize(width: kNumObjectsX, height: kNumObjectsY, depth: 1), threadsPerObjectThreadgroup: MTLSize(width: kMaxTotalThreadsPerObjectThreadgroup, height: 1, depth: 1), threadsPerMeshThreadgroup: MTLSize(width: kMaxTotalThreadsPerMeshThreadgroup, height: 1, depth: 1))

        renderEncoder.endEncoding()

        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
