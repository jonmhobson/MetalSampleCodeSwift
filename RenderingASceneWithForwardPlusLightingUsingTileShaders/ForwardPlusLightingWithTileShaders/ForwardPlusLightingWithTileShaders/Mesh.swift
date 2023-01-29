import MetalKit
import ModelIO

struct Submesh {
    let metalKitSubmesh: MTKSubmesh
    var textures: [MTLTexture?]

    static func createTextureFromMaterial(material: MDLMaterial,
                                          materialSemantic: MDLMaterialSemantic,
                                          textureLoader: MTKTextureLoader) -> MTLTexture {
        let propertiesWithSemantic = material.properties(with: materialSemantic)

        for property in propertiesWithSemantic where property.type == .string || property.type == .URL {
            // Load the textures with shader read using private storage
            let textureLoaderOptions: [MTKTextureLoader.Option: Any] = [
                MTKTextureLoader.Option.textureUsage : NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                MTKTextureLoader.Option.textureStorageMode : NSNumber(value: MTLStorageMode.private.rawValue),
                MTKTextureLoader.Option.SRGB: NSNumber(booleanLiteral: false)
            ]

            // First will interpret the string as a file path and attempt to load it with
            // MTKTextureLoader.newTextureWithContentsOfURL
            guard let urlString = {
                if property.type == .URL, let url = property.urlValue {
                    return url.absoluteString
                } else if let url = property.stringValue {
                    return "file://\(url)"
                } else {
                    return nil
                }
            }(),
            let textureURL = URL(string: urlString) else { fatalError() }
            // Attempt to load the texture from the file system
            if let texture = try? textureLoader.newTexture(URL: textureURL/*, options: textureLoaderOptions*/) {
                // If it found a texture, return it.
                return texture
            }

            // If MetalKit did not find a texture by interpreting the string as a path, interpret
            // the last component of the string as an asset catalog name and attempt to load it
            // with MTKTextureLoader.newTextureWithName
            if let lastComponent = urlString.components(separatedBy: "/").last, let texture = try? textureLoader.newTexture(name: lastComponent, scaleFactor: 1.0, bundle: nil, options: textureLoaderOptions) {
                // If there exists a texture with the string in the asset catalog...
                return texture
            }

            fatalError("Texture data for material property not found. \(materialSemantic) \(property.stringValue ?? "nil")")
        }

        fatalError("No appropriate material property from which to create texture")
    }

    init(modelIOSubmesh: MDLSubmesh,
         metalKitSubmesh: MTKSubmesh,
         textureLoader: MTKTextureLoader) {
        self.metalKitSubmesh = metalKitSubmesh
        let numTextures = Int(NumTextureIndices.rawValue)
        self.textures = (0..<numTextures).map { _ -> MTLTexture? in nil }

        self.textures[TextureIndexBaseColor.index] = Submesh.createTextureFromMaterial(
            material: modelIOSubmesh.material!,
            materialSemantic: .baseColor,
            textureLoader: textureLoader
        )

        self.textures[TextureIndexSpecular.index] = Submesh.createTextureFromMaterial(
            material: modelIOSubmesh.material!,
            materialSemantic: .specular,
            textureLoader: textureLoader
        )

        self.textures[TextureIndexNormal.index] = Submesh.createTextureFromMaterial(
            material: modelIOSubmesh.material!,
            materialSemantic: .tangentSpaceNormal,
            textureLoader: textureLoader
        )
    }
}

struct Mesh {
    let metalKitMesh: MTKMesh
    let submeshes: [Submesh]

    init(modelIOMesh: MDLMesh,
         vertexDescriptor: MDLVertexDescriptor,
         textureLoader: MTKTextureLoader,
         device: MTLDevice) {

        // Have ModelIO create the tangents from mesh texture coordinates and normals
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    normalAttributeNamed: MDLVertexAttributeNormal,
                                    tangentAttributeNamed: MDLVertexAttributeTangent)

        // Have ModelIO create bitangents from mesh texture coordinates and the newly created tangents
        modelIOMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                    tangentAttributeNamed: MDLVertexAttributeTangent,
                                    bitangentAttributeNamed: MDLVertexAttributeBitangent)

        // Apply the ModelIO vertex descriptor created to match the Metal vertex descriptor.
        // Assigning a new vertex descriptor to a ModelIO mesh performs a re-layout of the vertex
        // vertex data. In this case we created the ModelIO vertex descriptor so that the layout
        // of the vertices in the ModelIO mesh match the layout of vertices the Metal render pipeline
        // expects as input into its vertex shader.

        // Note that this re-layout operation can only be performed after tangents and
        // bitangents have been created. This is because ModelIO's addTangentBasis methods only work
        // with vertex data that is all in 32-bit floating-point. The vertex descriptor applied here can
        // change those floats into 16-bit floats or other types from which ModelIO cannot produce
        // tangents.
        modelIOMesh.vertexDescriptor = vertexDescriptor

        // Create the metalKit mesh which will contain the Metal buffer(s) with the mesh's vertex data
        // and submeshes with info to draw the mesh
        let metalKitMesh = try! MTKMesh(mesh: modelIOMesh, device: device)
        self.metalKitMesh = metalKitMesh

        // There should always be the same number of MetalKit submeshes in the MetalKit mesh as there
        // are ModelIO submeshes in the ModelIO mesh
        assert(metalKitMesh.submeshes.count == modelIOMesh.submeshes?.count)

        self.submeshes = metalKitMesh.submeshes.enumerated().map { (i, submesh) in
            Submesh(modelIOSubmesh: modelIOMesh.submeshes![i] as! MDLSubmesh,
                    metalKitSubmesh: submesh,
                    textureLoader: textureLoader)
        }
    }

    static func newMeshes(object: MDLObject,
                          vertexDescriptor: MDLVertexDescriptor,
                          textureLoader: MTKTextureLoader,
                          device: MTLDevice) -> [Mesh] {
        var newMeshes: [Mesh] = []

        // If this ModelIO object is a mesh object (not a camera, light, or something else)
        if let mesh = object as? MDLMesh {
            newMeshes.append(Mesh(modelIOMesh: mesh,
                                  vertexDescriptor: vertexDescriptor,
                                  textureLoader: textureLoader,
                                  device: device))
        }

        // Recursively traverse the ModelIO asset heirarchy to find ModelIO meshes that are children
        // of this ModelIO object and create app-specific Mesh objects from those ModelIO meshes
        for child in object.children.objects {
            let childMeshes = Mesh.newMeshes(object: child,
                                             vertexDescriptor: vertexDescriptor,
                                             textureLoader: textureLoader,
                                             device: device)
            newMeshes.append(contentsOf: childMeshes)
        }

        return newMeshes
    }

    static func newMeshes(url: URL,
                          vertexDescriptor: MDLVertexDescriptor,
                          device: MTLDevice) -> [Mesh] {
        // Create a MetalKit mesh buffer allocator so that ModelIO will load mesh data directly into
        // Metal buffers accessible by the GPU
        let bufferAllocator = MTKMeshBufferAllocator(device: device)

        // Use ModelIO to load the model file at the URL. This returns a ModelIO asset object, which
        // contains a hierarchy of ModelIO objects composing a "scene" described by the model file.
        // This hierarchy may include lights, cameras, but, most importantly, mesh and submesh data
        // that we'll render with Metal
        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: bufferAllocator)

        // Create a MetalKit texture loader to load material textures from files or the asset catalog
        // into Metal textures
        let textureLoader = MTKTextureLoader(device: device)

        var newMeshes: [Mesh] = []

        for object in asset.childObjects(of: MDLObject.self) {
            newMeshes.append(contentsOf: Mesh.newMeshes(object: object,
                                                        vertexDescriptor: vertexDescriptor,
                                                        textureLoader: textureLoader,
                                                        device: device))
        }

        return newMeshes
    }
}
