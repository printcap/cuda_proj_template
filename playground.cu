#include <cinttypes>
#include <iostream>

#include "cudautils.h"

template<typename T>
__global__ void
increment(T* array, T increment_val, unsigned num_elements) {
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < num_elements; idx += blockDim.x * gridDim.x) {
    array[idx] += increment_val;
  }
}

int main() {
  using ArrayType = int;
  constexpr size_t array_length = 1024 * 1024;
  constexpr ArrayType increment_val = 1;

  constexpr int device = 0;
  CUDA_CALL(cudaSetDevice(device));
  cudaDeviceProp device_props;
  CUDA_CALL(cudaGetDeviceProperties(&device_props, device));
  std::cout << "Device " << device << ": " << device_props.name << '\n';
  std::cout << "SM count: " << device_props.multiProcessorCount << '\n';
  std::cout << "Global memory: " << (device_props.totalGlobalMem >> 20) << " MiB\n";

  ArrayType *array;
  CUDA_CALL(cudaMallocManaged(&array, sizeof(ArrayType) * array_length));
  for (auto i = 0; i < array_length; i++) {
    array[i] = i;
  }

  dim3 blocks(2 * device_props.multiProcessorCount);
  dim3 threads(device_props.maxThreadsDim[0]);
  increment<<<blocks,threads>>>(array, increment_val, array_length);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  bool success = true;
  for (auto i = 0; success & (i < array_length); i++) {
    success = array[i] = (i + increment_val);
  }
  std::cout << (success ? "success" : "fail") << '\n';

  CUDA_CALL(cudaFree(array));
  return 0;
}
