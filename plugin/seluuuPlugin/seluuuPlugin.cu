/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


 #include "seluuuPlugin.h"
 #include <math.h>
 #include <cuda_fp16.h>


 template <typename T_DATA>
     __global__ void kernelCopy(
         int N,
         T_DATA* inputs,
         T_DATA* outputs
         )
 {
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < N){
         outputs[index] = inputs[index];
     }
     __syncthreads();
 }

constexpr float alpha = 1.6732632423543772848170429916717f;
constexpr float beta = 1.0507009873554804934193349852946f;

__global__ void kernel_selu(
    int N,
    const float *inputs,
    float* outputs
)
 {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N){
         if (inputs[index] > 0) {
            outputs[index] = beta * inputs[index];
         } else {
            outputs[index] = beta * alpha * (exp(inputs[index]) - 1);
         }
    }
 }

 int inference(
     int batchSize,
     int dataDim,
     float* inputs,
     float* outputs,
     cudaStream_t stream
) {
    int N = batchSize * dataDim;
    int N_blocks = N % 512 == 0 ? N / 512 : N / 512 + 1;

    kernel_selu<<<N_blocks, 512, 0, stream>>>(N, inputs, outputs);

     cudaError_t err = cudaGetLastError();
     if ( cudaSuccess != err )
     {
         fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
         return 1;
     }
     return 0;
}

 int SeluuuPlugin::enqueue(
     int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    return inference(batchSize, dataDim, (float*)inputs[0], (float*)outputs[0], stream);
}
