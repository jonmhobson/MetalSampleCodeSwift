#include <metal_stdlib>
using namespace metal;

// This is the C version of the add_arrays function below:

// void add_arrays(const float* inA,
//                 const float* inB,
//                 float* result,
//                 int length)
// {
//     for (int index = 0; index < length ; index++)
//     {
//         result[index] = inA[index] + inB[index];
//     }
// }

[[kernel]]
void add_arrays(device const float* inA,
                device const float* inB,
                device float* result,
                uint index [[thread_position_in_grid]]) {
    // The for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}
