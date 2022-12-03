import Foundation
import CoreGraphics
import ImageIO
import Metal

private func createCGImageFromFile(path: String) -> CGImage? {
    let url = URL(filePath: path)
    let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil)!
    let myImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)!
    return myImage
}

private struct RGB16Pixel {
    let r: Float16
    let g: Float16
    let b: Float16
}

private struct RGBA16Pixel {
    let rgb: RGB16Pixel
    let a: Float16 = 1.0
}

func TexturefromRadianceFile(filename: String, device: MTLDevice) -> MTLTexture? {

    // Validate the function inputs
    let subStrings = filename.components(separatedBy: ".")

    if subStrings[1] != "hdr" {
        return nil
    }

    let filePath = Bundle.main.path(forResource: subStrings[0], ofType: subStrings[1])!
    guard let loadedImage = createCGImageFromFile(path: filePath) else { fatalError() }

    let bpp = loadedImage.bitsPerPixel

    let sourceChannelCount = 3
    let bitsPerByte = 8
    let expectedBPP = MemoryLayout<UInt16>.stride * sourceChannelCount * bitsPerByte

    assert(bpp == expectedBPP)

    let width = loadedImage.width
    let height = loadedImage.height

    let pixelCount = width * height
    let dstChannelCount = 4

    let imgData = loadedImage.dataProvider!.data
    let srcData = CFDataGetBytePtr(imgData)!

    var dstData: [RGBA16Pixel] = []
    dstData.reserveCapacity(pixelCount)

    srcData.withMemoryRebound(to: RGB16Pixel.self, capacity: pixelCount) { ptr in
        var i = 0
        while i < pixelCount {
            dstData.append(RGBA16Pixel(rgb: ptr[i]))
            i += 1
        }
    }

    let texDesc = MTLTextureDescriptor()

    texDesc.pixelFormat = .rgba16Float
    texDesc.storageMode = .shared
    texDesc.width = width
    texDesc.height = height

    let texture = device.makeTexture(descriptor: texDesc)!
    let region = MTLRegionMake3D(0, 0, 0, width, height, 1)
    let bytesPerRow = MemoryLayout<Float16>.stride * dstChannelCount * width
    texture.replace(region: region, mipmapLevel: 0, withBytes: dstData, bytesPerRow: bytesPerRow)

    return texture
}

