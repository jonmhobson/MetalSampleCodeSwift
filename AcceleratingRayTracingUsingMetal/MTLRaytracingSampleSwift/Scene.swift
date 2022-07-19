import Foundation
import Metal

struct FaceMask: OptionSet {
    let rawValue: Int

    static let none: FaceMask = []
    static let negativeX = FaceMask(rawValue: (1 << 0))
    static let positiveX = FaceMask(rawValue: (1 << 1))
    static let negativeY = FaceMask(rawValue: (1 << 2))
    static let positiveY = FaceMask(rawValue: (1 << 3))
    static let negativeZ = FaceMask(rawValue: (1 << 4))
    static let positiveZ = FaceMask(rawValue: (1 << 5))
    static let all: FaceMask = [.negativeX, .positiveX,
                                .negativeY, .positiveY,
                                .negativeZ, .positiveZ]
}

struct BoundingBox {
    let min: MTLPackedFloat3
    let max: MTLPackedFloat3
}

class Geometry: NSObject {
    let device: MTLDevice
    var intersectionFunctionName: String? { nil }

    func clear() {}
    func uploadToBuffers() {}

    var geometryDescriptor: MTLAccelerationStructureGeometryDescriptor? {
        return nil
    }

    var resources: [MTLResource] {
        return []
    }

    init(device: MTLDevice) {
        self.device = device
    }
}

func getTriangleNormal(v0: vector_float3, v1: vector_float3, v2: vector_float3) -> vector_float3 {
    let e1 = normalize(v1 - v0)
    let e2 = normalize(v2 - v0)
    return cross(e1, e2)
}

class TriangleGeometry: Geometry {
    var vertexPositionBuffer: MTLBuffer? = nil
    var vertexNormalBuffer: MTLBuffer? = nil
    var vertexColorBuffer: MTLBuffer? = nil

    var vertices: [vector_float3] = []
    var normals: [vector_float3] = []
    var colors: [vector_float3] = []

    override func uploadToBuffers() {
        vertexPositionBuffer = device.makeBuffer(length: vertices.count * MemoryLayout<vector_float3>.stride, options: .storageModeShared)
        vertexNormalBuffer = device.makeBuffer(length: normals.count * MemoryLayout<vector_float3>.stride, options: .storageModeShared)
        vertexColorBuffer = device.makeBuffer(length: colors.count * MemoryLayout<vector_float3>.stride, options: .storageModeShared)

        memcpy(vertexPositionBuffer!.contents(), &vertices, vertexPositionBuffer!.length)
        memcpy(vertexNormalBuffer!.contents(), &normals, vertexNormalBuffer!.length)
        memcpy(vertexColorBuffer!.contents(), &colors, vertexColorBuffer!.length)
    }

    func addCube(faceMask: FaceMask,
                 color: vector_float3,
                 transform: matrix_float4x4,
                 inwardNormals: Bool) {
        var cubeVertices = [
            vector_float3(-0.5, -0.5, -0.5),
            vector_float3( 0.5, -0.5, -0.5),
            vector_float3(-0.5,  0.5, -0.5),
            vector_float3( 0.5,  0.5, -0.5),
            vector_float3(-0.5, -0.5,  0.5),
            vector_float3( 0.5, -0.5,  0.5),
            vector_float3(-0.5,  0.5,  0.5),
            vector_float3( 0.5,  0.5,  0.5)
        ]

        cubeVertices = cubeVertices.map { (transform * vector_float4($0.x, $0.y, $0.z, 1.0))[SIMD3(0, 1, 2)] }

        let cubeIndices: [[Int]] = [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6]
        ]

        for face in 0..<6 {
            if faceMask.contains(FaceMask(rawValue: 1 << face)) {
                addCubeFaceWith(cubeVertices: cubeVertices, color: color, i0: cubeIndices[face][0], i1: cubeIndices[face][1], i2: cubeIndices[face][2], i3: cubeIndices[face][3], inwardNormals: inwardNormals)
            }
        }
    }

    private func addCubeFaceWith(cubeVertices: [vector_float3],
                                 color: vector_float3,
                                 i0: Int, i1: Int, i2: Int, i3: Int,
                                 inwardNormals: Bool) {
        let v0 = cubeVertices[i0]
        let v1 = cubeVertices[i1]
        let v2 = cubeVertices[i2]
        let v3 = cubeVertices[i3]

        var n0 = getTriangleNormal(v0: v0, v1: v1, v2: v2)
        var n1 = getTriangleNormal(v0: v0, v1: v2, v2: v3)

        if inwardNormals {
            n0 = -n0
            n1 = -n1
        }

        vertices.append(contentsOf: [v0, v1, v2, v0, v2, v3])

        for _ in 0..<3 { normals.append(n0) }
        for _ in 0..<3 { normals.append(n1) }
        for _ in 0..<6 { colors.append(color) }
    }

    override var geometryDescriptor: MTLAccelerationStructureGeometryDescriptor? {
        // Metal represents each piece of geometry in an acceleration structure using
        // a geometry descriptor. The sample uses a triangle geometry descriptor to represent
        // triangle geometry. Each triangle geometry descriptor can have its own
        // vertex buffer, index buffer, and triangle count. The sample uses a single geometry
        // descriptor since it already packed all of the vertex data into a single buffer.
        let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        descriptor.vertexBuffer = vertexPositionBuffer
        descriptor.vertexStride = MemoryLayout<vector_float3>.stride
        descriptor.triangleCount = vertices.count / 3
        return descriptor
    }

    override var resources: [MTLResource] {
        return [vertexNormalBuffer!, vertexColorBuffer!]
    }
}

class SphereGeometry: Geometry {
    var sphereBuffer: MTLBuffer? = nil
    var boundingBoxBuffer: MTLBuffer? = nil

    var spheres: [Sphere] = []

    func addSphere(origin: vector_float3,
                   radius: Float,
                   color: vector_float3) {
        spheres.append(Sphere(origin: origin, radius: radius, color: color))
    }

    override func uploadToBuffers() {
        sphereBuffer = device.makeBuffer(length: spheres.count * MemoryLayout<Sphere>.stride, options: .storageModeShared)
        boundingBoxBuffer = device.makeBuffer(length: spheres.count * MemoryLayout<BoundingBox>.stride, options: .storageModeShared)

        var boundingBoxes: [BoundingBox] = []

        // Geometry types that use custom instersection functions provide bounding boxes that enclose
        // each primitive. Metal invokes the intersection function whenever a ray potentially intersects
        // one of these bounding boxes.
        spheres.forEach { sphere in
            let bounds = BoundingBox(min: MTLPackedFloat3Make(
                                            sphere.origin.x - sphere.radius,
                                            sphere.origin.y - sphere.radius,
                                            sphere.origin.z - sphere.radius),
                                     max: MTLPackedFloat3Make(
                                            sphere.origin.x + sphere.radius,
                                            sphere.origin.y + sphere.radius,
                                            sphere.origin.z + sphere.radius))
            boundingBoxes.append(bounds)
        }

        memcpy(sphereBuffer!.contents(), &spheres, sphereBuffer!.length)
        memcpy(boundingBoxBuffer!.contents(), &boundingBoxes, boundingBoxBuffer!.length)
    }

    override var geometryDescriptor: MTLAccelerationStructureGeometryDescriptor? {
        let descriptor = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
        descriptor.boundingBoxBuffer = boundingBoxBuffer
        descriptor.boundingBoxCount = spheres.count
        return descriptor
    }

    override var resources: [MTLResource] {
        return [ sphereBuffer! ]
    }

    override var intersectionFunctionName: String? {
        return "sphereIntersectionFunction"
    }
}

class GeometryInstance {
    let geometry: Geometry
    let transform: matrix_float4x4
    var mask: Int32

    init(geometry: Geometry, transform: matrix_float4x4, mask: Int32) {
        self.geometry = geometry
        self.transform = transform
        self.mask = mask
    }
}

class Scene {
    let device: MTLDevice

    var geometries: [Geometry] = []
    var instances: [GeometryInstance] = []
    var lights: [AreaLight] = []

    let cameraPosition = vector_float3(0, 1, 10.0)
    let cameraTarget = vector_float3(0, 1, 0)
    let cameraUp = vector_float3(0, 1, 0)

    init(device: MTLDevice) {
        self.device = device

        let lightMesh = TriangleGeometry(device: device)
        geometries.append(lightMesh)

        var transform = matrix4x4_translation(tx: 0, ty: 1, tz: 0) *
                        matrix4x4_scale(sx: 0.5, sy: 1.98, sz: 0.5)

        lightMesh.addCube(faceMask: .positiveY, color: vector_float3(1, 1, 1), transform: transform, inwardNormals: true)

        let geometryMesh = TriangleGeometry(device: device)
        geometries.append(geometryMesh)

        transform = matrix4x4_translation(tx: 0, ty: 1, tz: 0) *
                    matrix4x4_scale(sx: 2, sy: 2, sz: 2)

        // Add the top, bottom, and back walls.
        geometryMesh.addCube(faceMask: [.negativeY, .positiveY, .negativeZ], color: vector3(0.725, 0.71, 0.68),
                             transform: transform, inwardNormals: true)

        // Add the left wall.
        geometryMesh.addCube(faceMask: [.negativeX], color: vector3(0.63, 0.065, 0.05),
                             transform: transform, inwardNormals: true)

        // Add the right wall.
        geometryMesh.addCube(faceMask: [.positiveX], color: vector3(0.14, 0.45, 0.091),
                             transform: transform, inwardNormals: true)

        transform = matrix4x4_translation(tx: -0.335, ty: 0.6, tz: -0.29) *
                    matrix4x4_rotation(radians: 0.3, axis: vector3(0, 1, 0)) *
                    matrix4x4_scale(sx: 0.6, sy: 1.2, sz: 0.6)

        // Add the tall box.
        geometryMesh.addCube(faceMask: .all, color: vector3(0.725, 0.71, 0.68),
                             transform: transform, inwardNormals: false)

        let sphereGeometry = SphereGeometry(device: device)
        geometries.append(sphereGeometry)
        sphereGeometry.addSphere(origin: vector3(0.3275, 0.3, 0.3725),
                                 radius: 0.3,
                                 color: vector3(0.725, 0.71, 0.68))

        // Create nine instances of the scene
        for y in -1...1 {
            for x in -1...1 {
                let transform = matrix4x4_translation(tx: Float(x) * 2.5, ty: Float(y) * 2.5, tz: 0)

                // Create an instance of the light
                let lightMeshInstance = GeometryInstance(geometry: lightMesh, transform: transform, mask: GEOMETRY_MASK_LIGHT)
                instances.append(lightMeshInstance)

                // Create an instance of the Cornell Box.
                let geometryMeshInstance = GeometryInstance(geometry: geometryMesh, transform: transform, mask: GEOMETRY_MASK_TRIANGLE)
                instances.append(geometryMeshInstance)

                // Create an instance of the sphere.
                let sphereGeometryInstance = GeometryInstance(geometry: sphereGeometry, transform: transform, mask: GEOMETRY_MASK_SPHERE)
                instances.append(sphereGeometryInstance)

                let light = AreaLight(position: vector3(Float(x) * 2.5,Float(y) * 2.5 + 1.98, 0),
                                      forward: vector3(0, -1, 0),
                                      right: vector3(0.25, 0.0, 0.0),
                                      up: vector3(0, 0, 0.25),
                                      color: vector3(Float.random(in: 0...4.0),
                                                     Float.random(in: 0...4.0),
                                                     Float.random(in: 0...4.0)))

                lights.append(light)
            }
        }
    }

    var lightBuffer: MTLBuffer? = nil

    func uploadToBuffers() {
        geometries.forEach { $0.uploadToBuffers() }

        lightBuffer = device.makeBuffer(length: lights.count * MemoryLayout<AreaLight>.stride, options: .storageModeShared)

        guard let lightBuffer = lightBuffer else { return }
        memcpy(lightBuffer.contents(), &lights, lightBuffer.length)
    }
}
