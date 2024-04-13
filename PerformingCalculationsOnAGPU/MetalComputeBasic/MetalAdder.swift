import Foundation
import Metal

// The number of floats in each array.
fileprivate let arrayLength = 1 << 24

struct MetalAdder {
    private let device: any MTLDevice
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    private let addFunctionPSO: any MTLComputePipelineState
    // The command queue used to pass commands to the device.
    private let commandQueue: any MTLCommandQueue

    // Buffers to hold data.
    private let bufferA: MetalBuffer<Float>
    private let bufferB: MetalBuffer<Float>
    private let resultBuffer: MetalBuffer<Float>

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Could not find any metal devices")
        }

        do {
            // Load the shader files with a .metal file extension in the project
            let defaultLibrary = try device.makeDefaultLibrary(bundle: Bundle.main)
            let addFunction = try defaultLibrary.makeFunction(
                name: "add_arrays",
                constantValues: MTLFunctionConstantValues()
            )

            // Create a compute pipeline state object.
            self.addFunctionPSO = try device.makeComputePipelineState(function: addFunction)
        } catch {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            fatalError(error.localizedDescription)
        }

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }

        guard let bufferA = MetalBuffer<Float>(device: device, length: arrayLength),
              let bufferB = MetalBuffer<Float>(device: device, length: arrayLength),
              let results = MetalBuffer<Float>(device: device, length: arrayLength) else {
            fatalError("Failed to create buffers")
        }

        self.device = device
        self.commandQueue = commandQueue
        self.bufferA = bufferA
        self.bufferB = bufferB
        self.resultBuffer = results
    }

    func prepareData() {
        bufferA.fillWithUnitRandomValues()
        bufferB.fillWithUnitRandomValues()
    }

    func sendComputeCommand() {
        // Create a command buffer to hold commands, and start a compute pass
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError()
        }

        encodeAddCommand(computeEncoder: computeEncoder)

        // End the compute pass
        computeEncoder.endEncoding()

        // Execute the command.
        commandBuffer.commit()

        // Normally, you want to do other work in your app while the GPU is running,
        // but in this example, the code simply blocks until the calculation is complete.
        commandBuffer.waitUntilCompleted()

        verifyResults()
    }

    private func encodeAddCommand(computeEncoder: any MTLComputeCommandEncoder) {
        // Encode the pipeline state object and its parameters.
        computeEncoder.setComputePipelineState(addFunctionPSO)
        computeEncoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)

        let gridSize = MTLSize(
            width: arrayLength,
            height: 1,
            depth: 1
        )

        // Calculate a threadgroup size.
        let threadgroupSize = MTLSize(
            width: min(addFunctionPSO.maxTotalThreadsPerThreadgroup, arrayLength),
            height: 1,
            depth: 1
        )

        // Encode the compute command.
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    private func verifyResults() {
        let results = resultBuffer.contents
        let pointerA = bufferA.contents
        let pointerB = bufferB.contents

        // Swift's For-In loop is quite slow in Debug over 16.7 million elements,
        // so use a while loop instead (they perform identically in Release)
        var i = 0
        while i < arrayLength {
            if results[i] != (pointerA[i] + pointerB[i]) {
                fatalError("Compute ERROR: index=\(i) result=\(results[i]) vs a+b=\(pointerA[i] + pointerB[i])")
            }
            i += 1
        }

        print("Compute results as expected")
    }
}
