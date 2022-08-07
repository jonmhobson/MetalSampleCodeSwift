import Metal

class EventWrapper {
    let device: MTLDevice
    let event: MTLEvent
    var signalCounter: UInt64

    init(device: MTLDevice) {
        self.device = device
        self.event = {
            guard let event = device.makeEvent() else { fatalError("Failed to create MTLEvent")}
            return event
        }()
        self.signalCounter = 0
    }

    func wait(commandBuffer: MTLCommandBuffer) {
        commandBuffer.encodeWaitForEvent(event, value: signalCounter)
    }

    func signal(commandBuffer: MTLCommandBuffer) {
        signalCounter += 1
        commandBuffer.encodeSignalEvent(event, value: signalCounter)
    }
}
