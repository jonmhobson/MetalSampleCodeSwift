import Foundation
import MetalKit

private let maxFramesInFlight = 3
private let numTriangles = 50

@MainActor
final class Renderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState

    private let inFlightSemaphore = DispatchSemaphore(value: maxFramesInFlight)
    private var vertexBuffers: [MTLBuffer] = []

    private var totalVertexCount: Int = 0
    private var currentBufferIndex = 0

    private var viewportSize: simd_uint2 = [0, 0]
    private var triangles: [Triangle] = []

    private var wavePosition: Float = 0.0

    init(device: any MTLDevice, pixelFormat: MTLPixelFormat) {
        guard let commandQueue = device.makeCommandQueue() else { fatalError() }

        self.device = device
        self.commandQueue = commandQueue

        guard let defaultLibrary = device.makeDefaultLibrary(),
              let vertexFunction = defaultLibrary.makeFunction(name: "vertexShader"),
              let fragmentFunction = defaultLibrary.makeFunction(name: "fragmentShader") else { fatalError() }

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.label = "MyPipeline"
        pipelineStateDescriptor.vertexFunction = vertexFunction
        pipelineStateDescriptor.fragmentFunction = fragmentFunction
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = pixelFormat
        pipelineStateDescriptor.vertexBuffers[Int(AAPLVertexInputIndexVertices.rawValue)].mutability = .immutable

        self.pipelineState = {
            do {
                return try device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
            } catch {
                fatalError("Failed to create pipeline state: \(error)")
            }
        }()

        super.init()

        generateTriangles()

        let triangleVertexCount = Triangle.vertices.count
        self.totalVertexCount = triangleVertexCount * numTriangles
        let triangleVertexBufferSize = MemoryLayout<AAPLVertex>.stride * totalVertexCount

        for i in 0..<maxFramesInFlight {
            if let buffer = device.makeBuffer(length: triangleVertexBufferSize, options: .storageModeShared) {
                buffer.label = "Vertex buffer \(i)"
                vertexBuffers.append(buffer)
            }
        }
    }

    func generateTriangles() {
        // Array of colors.
        let colors: [simd_float4] = [
            [1, 0, 0, 1], // Red
            [0, 1, 0, 1], // Green
            [0, 0, 1, 1], // Blue
            [1, 0, 1, 1], // Magenta
            [0, 1, 1, 1], // Cyan
            [1, 1, 0, 1], // Yellow
        ]

        let numColors = colors.count

        // Horizontal spacing between each triangle.
        let horizontalSpacing: Float = 16.0

        triangles = (0..<numTriangles).map { t in
            let half = Float(numTriangles) * 0.5
            let x = Float(t) - half
            return Triangle(position: [x * horizontalSpacing, 0], color: colors[t % numColors])
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = [UInt32(size.width), UInt32(size.height)]
    }

    static private let triangleVertices: [AAPLVertex] = [
        AAPLVertex(position: [ 250, -250], color: [1, 0, 0, 1]),
        AAPLVertex(position: [-250, -250], color: [0, 1, 0, 1]),
        AAPLVertex(position: [   0,  250], color: [0, 0, 1, 1])
    ]

    func updateState() {
        let waveMagnitude: Float = 128.0
        let waveSpeed: Float = 0.05

        wavePosition += waveSpeed

        let triangleVertices = Triangle.vertices
        let triangleVertexCount = triangleVertices.count

        let currentVertices = vertexBuffers[currentBufferIndex].contents().bindMemory(to: AAPLVertex.self, capacity: triangleVertexCount)

        for triangle in 0..<numTriangles {
            var trianglePosition = triangles[triangle].position
            trianglePosition.y = (sin(trianglePosition.x / waveMagnitude + wavePosition) * waveMagnitude)

            triangles[triangle].position = trianglePosition

            for vertex in 0..<triangleVertexCount {
                let currentVertex = vertex + (triangle * triangleVertexCount)
                currentVertices[currentVertex].position = triangleVertices[vertex].position + triangles[triangle].position
                currentVertices[currentVertex].color = triangles[triangle].color
            }
        }
    }

    func draw(in view: MTKView) {
        inFlightSemaphore.wait()

        currentBufferIndex = (currentBufferIndex + 1) % maxFramesInFlight

        updateState()

        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }

        commandBuffer.label = "MyCommand"
        renderEncoder.label = "MyRenderEncoder"

        renderEncoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                              width: Double(viewportSize.x),
                                              height: Double(viewportSize.y),
                                              znear: 0, zfar: 0))

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setVertexBuffer(vertexBuffers[currentBufferIndex], offset: 0, index: Int(AAPLVertexInputIndexVertices.rawValue))

        renderEncoder.setVertexBytes(&viewportSize,
                                     length: MemoryLayout<simd_uint2>.stride,
                                     index: Int(AAPLVertexInputIndexViewportSize.rawValue))

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: totalVertexCount)

        renderEncoder.endEncoding()

        if let drawable = view.currentDrawable {
            commandBuffer.present(drawable)
        }

        commandBuffer.addCompletedHandler { [inFlightSemaphore] _ in
            inFlightSemaphore.signal()
        }

        commandBuffer.commit()
    }
}
