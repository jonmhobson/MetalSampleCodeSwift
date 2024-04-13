import Metal

/// A typesafe wrapper for a MTLBuffer
struct MetalBuffer<T> {
    let buffer: any MTLBuffer
    let length: Int

    init?(device: any MTLDevice, length: Int) {
        guard let buffer = device.makeBuffer(
            length: length * MemoryLayout<T>.stride,
            options: .storageModeShared
        ) else { return nil }
        self.buffer = buffer
        self.length = length
    }

    var contents: UnsafeMutablePointer<T> {
        buffer.contents().bindMemory(to: T.self, capacity: length)
    }
}

extension MetalBuffer where T == Float {
    func fillWithUnitRandomValues() {
        let p = contents
        // Use a while loop because we're looping over so many elements that the overhead of
        // For-in is significant in Debug builds.
        var i = 0
        while i < length {
            p[i] = Float(drand48())
            i += 1
        }
    }
}
