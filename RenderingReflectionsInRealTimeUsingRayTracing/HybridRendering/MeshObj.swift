import MetalKit
import ModelIO

struct SubmeshObj {
    let metalKitSubmesh: MTKSubmesh
    let textures: [MTLTexture]

    static func createMetalTexture(from material: MDLMaterial,
                                   materialSemantic: MDLMaterialSemantic,
                                   defaultPropertyType: MDLMaterialPropertyType,
                                   textureLoader: MTKTextureLoader) -> MTLTexture {

        let propertiesWithSemantic = material.properties(with: materialSemantic)

        for property in propertiesWithSemantic {
            assert(property.semantic == materialSemantic)

            if property.type != .string { continue }

            // Load textures with TextureUsageShaderRead and StorageModePrivate
            let textureLoaderOptions: [MTKTextureLoader.Option: Any] = [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
            ]

            // Interpret the string as a file path and attempt to load it with an MTKTextureLoader
            let url = property.urlValue
            var urlString = ""

            if property.type == .URL {
                urlString = url!.absoluteString
            } else {
                urlString = "file://\(property.stringValue!)"
            }

            let textureUrl = URL(string: urlString)!

            // Attempt to load the texture from the file system.
            if let texture = try? textureLoader.newTexture(URL: textureUrl, options: textureLoaderOptions) {
                // If the texture loader finds a texture for a material using the string as a file path
                // name, return it.
                return texture
            }

            // If the texture loader doesn't find a texture by interpreting the URL as a path,
            // interpret the string as an asset catalog name and attempt to load it with
            // newTexture(name:
            let lastComponent = property.stringValue!.components(separatedBy: "/").last!

            if let texture = try? textureLoader.newTexture(name: lastComponent, scaleFactor: 1.0, bundle: nil, options: textureLoaderOptions) {
                // If model i/o finds a texture by interpreting the URL as an asset
                // catalog name, return it.
                return texture
            }

            // If the texture loader doesn't find a texture by interpreting it as a file path or
            // as an asset name in the asset catalog, something is wrong. Perhaps the file is
            // missing or misnamed in the asset catalog, model/material file, or file system.

            // Depending on the implementation of the Metal render pipeline with this submesh,
            // the system can handle this condition more gracefully. The app can load a dummy
            // texture that looks OK when set with the pipeline, or ensure that the pipeline
            // rendering this submesh doesn't require a material with this property.
            fatalError("Texture not found")
        }

        fatalError("Texture not found")
    }

    init(modelIOSubmesh: MDLSubmesh, metalKitSubmesh: MTKSubmesh, textureLoader: MTKTextureLoader) {
        self.metalKitSubmesh = metalKitSubmesh

        var textures: [MTLTexture?] = Array<MTLTexture?>(repeating: nil, count: Int(MaterialTextureCount.rawValue))

        // Set each index in the array with the appropriate material semantic specified in the
        // submesh's material proper
        textures[Int(TextureIndexBaseColor.rawValue)] = Self.createMetalTexture(from: modelIOSubmesh.material!,
                                                                                materialSemantic: .baseColor,
                                                                                defaultPropertyType: .float3,
                                                                                textureLoader: textureLoader)

        textures[Int(TextureIndexMetallic.rawValue)] = Self.createMetalTexture(from: modelIOSubmesh.material!,
                                                                               materialSemantic: .metallic,
                                                                               defaultPropertyType: .float3,
                                                                               textureLoader: textureLoader)

        textures[Int(TextureIndexRoughness.rawValue)] = Self.createMetalTexture(from: modelIOSubmesh.material!,
                                                                                materialSemantic: .roughness,
                                                                                defaultPropertyType: .float3,
                                                                                textureLoader: textureLoader)

        textures[Int(TextureIndexNormal.rawValue)] = Self.createMetalTexture(from: modelIOSubmesh.material!,
                                                                             materialSemantic: .tangentSpaceNormal,
                                                                             defaultPropertyType: .none,
                                                                             textureLoader: textureLoader)

        textures[Int(TextureIndexAmbientOcclusion.rawValue)] = Self.createMetalTexture(from: modelIOSubmesh.material!,
                                                                                       materialSemantic: .ambientOcclusion,
                                                                                       defaultPropertyType: .none,
                                                                                       textureLoader: textureLoader)

        self.textures = textures.compactMap { $0 }
    }
}

struct MeshObj {
    let metalKitMesh: MTKMesh
    let submeshes: [SubmeshObj]

    init?(modelIOMesh: MDLMesh, vertexDescriptor: MDLVertexDescriptor, textureLoader: MTKTextureLoader, device: MTLDevice) {
        modelIOMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.98)

        // Have model IO create the tangents from the mesh texture coordinates and normals.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    normalAttributeNamed: MDLVertexAttributeNormal,
                                    tangentAttributeNamed: MDLVertexAttributeTangent)

        // Have model IO create bitangentss from the mesh texture coordinates and the newly created tangents.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    tangentAttributeNamed: MDLVertexAttributeTangent,
                                    bitangentAttributeNamed: MDLVertexAttributeBitangent)

        // Assigning a new vertex descriptor to a Model I/O mesh performs a relayout of the vertex
        // data. In this case, the renderer creates the Model IO vertex descriptor so that the
        // layout of the vertices in the ModelIO mesh match the layout of vertices that the Metal render
        // pipeline expects as input into its vertex shader.

        // Note: Model I/O must create tangents and bitangents (as done above) before this
        // relayout occurs.
        // Model I/O's addTangentBasis methods only work with vertex data that is all
        // in 32 bit floating point. Applying the vertex descriptor changes those floats
        // into 16-bit floating point or other types from which Model IO can't produce tangents.
        modelIOMesh.vertexDescriptor = vertexDescriptor

        // Create the MetalKit mesh, which contains the Metal buffers with the mesh's vertex data
        // and submeshes with data to draw the mesh.
        let metalKitMesh = try! MTKMesh(mesh: modelIOMesh, device: device)
        self.metalKitMesh = metalKitMesh

        // A MetalKit mesh needs to always have the same number of MetalKit submeshes
        // as the model IO mesh has submeshes.
        assert(metalKitMesh.submeshes.count == modelIOMesh.submeshes?.count ?? 0)

        // Create an array to hold this MeshObj's SubmeshObj objects.
        var submeshes: [SubmeshObj] = []

        // Create a SubmeshObj for each submesh and add it to the submesh's array
        for index in 0..<metalKitMesh.submeshes.count {
            // Create an app-specific submesh to hold the MetalKit submesh.
            let submesh = SubmeshObj(modelIOSubmesh: modelIOMesh.submeshes![index] as! MDLSubmesh,
                                     metalKitSubmesh: metalKitMesh.submeshes[index],
                                     textureLoader: textureLoader)
            submeshes.append(submesh)
        }

        self.submeshes = submeshes
    }

    init?(modelIOMesh: MDLMesh, material: MDLMaterial, device: MTLDevice, vertexDescriptor: MDLVertexDescriptor) {
        // Assigning a new vertex descriptor to a Model I/O mesh performs a relayout of
        // the vertex data.
        modelIOMesh.vertexDescriptor = vertexDescriptor

        // Create the MetalKit mesh that contains the Metal buffers with the mesh's
        // vertex data and submeshes with data to draw the mesh.
        let metalKitMesh = try! MTKMesh(mesh: modelIOMesh, device: device)

        self.metalKitMesh = metalKitMesh

        let textureLoader = MTKTextureLoader(device: device)

        var submeshes: [SubmeshObj] = []

        for i in 0..<metalKitMesh.submeshes.count {
            if let submesh = modelIOMesh.submeshes?[i] as? MDLSubmesh {
                submesh.material = material
                let submesh = SubmeshObj(modelIOSubmesh: submesh,
                                         metalKitSubmesh: metalKitMesh.submeshes[i],
                                         textureLoader: textureLoader)

                submeshes.append(submesh)
            }
        }

        self.submeshes = submeshes
    }

    static func newMeshes(from object: MDLObject, vertexDescriptor: MDLVertexDescriptor,
                          textureLoader: MTKTextureLoader, device: MTLDevice) -> [MeshObj] {
        var newMeshes: [MeshObj] = []

        // If this model IO object is a mesh object (not a camera, light, or something else),
        // create an app-specific MeshObj for it.
        if let mesh = object as? MDLMesh {
            if let newMesh = MeshObj(modelIOMesh: mesh, vertexDescriptor: vertexDescriptor, textureLoader: textureLoader, device: device) {
                newMeshes.append(newMesh)
            }
        }

        // Recursively traverse the Model I/O asset hierarchy to find nodes that are
        // Model I/O meshes and create app-specific MeshObj objects from those meshes.
        for child in object.children.objects {
            let childMeshes: [MeshObj] = Self.newMeshes(from: child, vertexDescriptor: vertexDescriptor, textureLoader: textureLoader, device: device)

            newMeshes.append(contentsOf: childMeshes)
        }

        return newMeshes
    }

    static func newMeshes(from url: URL, modelIOVertexDescriptor: MDLVertexDescriptor, metalDevice: MTLDevice) -> [MeshObj] {
        // Create a MetalKit mesh buffer allocator so that Model I/O loads mesh data directly into
        // Metal buffers accessible by the GPU
        let bufferAllocator = MTKMeshBufferAllocator(device: metalDevice)

        // Use ModelIO to load the model file at the URL. This returns a model I/O asset
        // object, which contains a heirarchy of ModelIO objects composing a "scene" that
        // the model file describes. This heirarchy may include lights and cameras, but,
        // most importantly, mesh and submesh data that Metal renders.
        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: bufferAllocator)

        // Create a MetalKit texture loader to load material textures from files or the asset catalog
        // into Metal Textures.
        let textureLoader = MTKTextureLoader(device: metalDevice)

        var newMeshes: [MeshObj] = []

        // Traverse the Model I/O asset hierarchy to find Model I/O meshes and create app-specific
        // MeshObj objects from those Model I/O meshes.
        for object in asset.childObjects(of: MDLObject.self) {
            let assetMeshes: [MeshObj] = MeshObj.newMeshes(from: object, vertexDescriptor: modelIOVertexDescriptor,
                                                           textureLoader: textureLoader, device: metalDevice)

            newMeshes.append(contentsOf: assetMeshes)
        }

        return newMeshes
    }

    init(skyboxMeshOnDevice device: MTLDevice, vertexDescriptor: MDLVertexDescriptor) {
        let bufferAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh = MDLMesh.newEllipsoid(withRadii: [200, 200, 200],
                                           radialSegments: 10,
                                           verticalSegments: 10,
                                           geometryType: .triangles,
                                           inwardNormals: true,
                                           hemisphere: false,
                                           allocator: bufferAllocator)

        mdlMesh.vertexDescriptor = vertexDescriptor

        let mtkMesh = try! MTKMesh(mesh: mdlMesh, device: device)

        self.metalKitMesh = mtkMesh
        self.submeshes = []
    }

    init(sphereWithRadius radius: Float, modelIOVertexDescriptor vertexDescriptor: MDLVertexDescriptor, metalDevice device: MTLDevice) {
        let bufferAllocator = MTKMeshBufferAllocator(device: device)

        let modelIOMesh = MDLMesh.newEllipsoid(withRadii: [radius, radius, radius],
                                               radialSegments: 20, verticalSegments: 20,
                                               geometryType: .triangles,
                                               inwardNormals: false, hemisphere: false, allocator: bufferAllocator)

        // Model I/O creates the tangents from the mesh's texture coordinates and normals.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    normalAttributeNamed: MDLVertexAttributeNormal,
                                    tangentAttributeNamed: MDLVertexAttributeTangent)

        // Model I/O creates bitangents from the mesh's texture coordinates and
        // the newly created tangents.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    tangentAttributeNamed: MDLVertexAttributeTangent,
                                    bitangentAttributeNamed: MDLVertexAttributeBitangent)

        modelIOMesh.vertexDescriptor = vertexDescriptor

        let material = MDLMaterial()

        // The texture strings that reference the contents of 'Assets.xcassets' file.
        material.setProperty(MDLMaterialProperty(name: "baseColor", semantic: .baseColor, string: "white"))
        material.setProperty(MDLMaterialProperty(name: "metallic", semantic: .metallic, string: "white"))
        material.setProperty(MDLMaterialProperty(name: "roughness", semantic: .roughness, string: "black"))
        material.setProperty(MDLMaterialProperty(name: "tangentNormal", semantic: .tangentSpaceNormal, string: "BodyNormalMap"))
        material.setProperty(MDLMaterialProperty(name: "ao", semantic: .ambientOcclusion, string: "white"))

        self.init(modelIOMesh: modelIOMesh,
                  material: material,
                  device: device,
                  vertexDescriptor: vertexDescriptor)!
    }

    init(planeWithDimensions dimensions: SIMD2<Float>, modelIOVertexDescriptor vertexDescriptor: MDLVertexDescriptor, metalDevice device: MTLDevice) {
        let bufferAllocator = MTKMeshBufferAllocator(device: device)

        let modelIOMesh = MDLMesh.newPlane(withDimensions: dimensions, segments: [100, 100], geometryType: .triangles, allocator: bufferAllocator)

        // Model I/O creates the tangents from the mesh's texture coordinates and normals.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    normalAttributeNamed: MDLVertexAttributeNormal,
                                    tangentAttributeNamed: MDLVertexAttributeTangent)

        // Model I/O creates bitangents from the mesh's texture coordinates and
        // the newly created tangents.
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    tangentAttributeNamed: MDLVertexAttributeTangent,
                                    bitangentAttributeNamed: MDLVertexAttributeBitangent)

        modelIOMesh.vertexDescriptor = vertexDescriptor

        // Repeat the floor texture coordinates 20 times over.
        let kFloorRepeat: Float = 20.0
        let texcoords = modelIOMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate)!
        let map = texcoords.map
        let uv = map.bytes.bindMemory(to: SIMD2<Float>.self, capacity: 1)
        for i in 0..<(texcoords.bufferSize / MemoryLayout<SIMD2<Float>>.stride) {
            uv[i].x *= kFloorRepeat
            uv[i].y *= kFloorRepeat
        }

        let material = MDLMaterial()
        material.setProperty(MDLMaterialProperty(name: "baseColor", semantic: .baseColor, string: "checkerboard_gray"))
        material.setProperty(MDLMaterialProperty(name: "metallic", semantic: .metallic, string: "white"))
        material.setProperty(MDLMaterialProperty(name: "roughness", semantic: .roughness, string: "black"))
        material.setProperty(MDLMaterialProperty(name: "tangentNormal", semantic: .tangentSpaceNormal, string: "black"))
        material.setProperty(MDLMaterialProperty(name: "ao", semantic: .ambientOcclusion, string: "white"))

        self.init(modelIOMesh: modelIOMesh, material: material, device: device, vertexDescriptor: vertexDescriptor)!
    }
}
