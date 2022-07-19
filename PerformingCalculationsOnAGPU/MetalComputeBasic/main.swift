import Foundation
import Metal

// Helper extension for filling a buffer pointed to by an UnsafeMutablePointer with random values between 0.0 and 1.0
extension UnsafeMutablePointer where Pointee == Float {
    func fillWithUnitRandomValues(count: Int) {
        var i = 0
        var seed: UInt32 = 1
        // Swift only has fairly expensive cryptographically secure random number generators,
        // which is overkill for our needs.
        // All the Obj-C Metal samples usually use rand() which is essentially this algorithm
        while i < count {
            seed = seed &* 1103515245 &+ 12345
            self[i] = Float((seed / 65536) % 32768) / Float(32768.0)
            i += 1
        }
    }
}

let arrayLength = 1 << 24
let bufferSize = arrayLength * MemoryLayout<Float>.stride

// MARK: INIT
guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
guard let defaultLibrary = device.makeDefaultLibrary() else {
    fatalError("Failed to find the default library.")
}

guard let addFunction = defaultLibrary.makeFunction(name: "add_arrays") else {
    fatalError("Failed to find the adder function.")
}

let addFunctionPipelineStateObject: MTLComputePipelineState = {
    do {
        return try device.makeComputePipelineState(function: addFunction)
    } catch {
        fatalError("Failed to create pipeline state object, error \(error)")
    }
}()

guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create a command queue.")
}

// MARK: PREPARE DATA
guard let bufferA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
      let bufferB = device.makeBuffer(length: bufferSize, options: .storageModeShared),
      let bufferResult = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
    fatalError()
}

let pointerA = bufferA.contents().bindMemory(to: Float.self, capacity: arrayLength)
let pointerB = bufferB.contents().bindMemory(to: Float.self, capacity: arrayLength)
let results = bufferResult.contents().bindMemory(to: Float.self, capacity: arrayLength)

pointerA.fillWithUnitRandomValues(count: arrayLength)
pointerB.fillWithUnitRandomValues(count: arrayLength)

// MARK: SEND COMPUTE COMMAND
guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { fatalError() }

computeEncoder.setComputePipelineState(addFunctionPipelineStateObject)
computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
computeEncoder.setBuffer(bufferResult, offset: 0, index: 2)

// The number of threads per grid.
let gridSize = MTLSize(width: arrayLength, height: 1, depth: 1)

// The number of threads per threadGroup
let threadgroupSize = MTLSize(width: min(addFunctionPipelineStateObject.maxTotalThreadsPerThreadgroup, arrayLength),
                              height: 1, depth: 1)

// Encode the compute command.
computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
computeEncoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// MARK: VERIFY RESULTS
private var i = 0
while i < arrayLength { // Use a while loop because it's far faster than for..in in debug in Swift
    if results[i] != (pointerA[i] + pointerB[i]) {
        fatalError("Compute ERROR: index=\(i) result=\(results[i]) vs a+b=\(pointerA[i] + pointerB[i])")
    }
    i += 1
}

print("Compute results as expected")
