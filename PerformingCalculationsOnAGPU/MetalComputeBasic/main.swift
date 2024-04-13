import Foundation
import Metal

let adder = MetalAdder()

adder.prepareData()
adder.sendComputeCommand()
