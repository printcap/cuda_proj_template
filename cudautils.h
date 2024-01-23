#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <sstream>

#define CUDA_CALL(call)                         \
do {                                            \
  cudaError_t status = call;                    \
  if (status != cudaSuccess) {                  \
    std::stringstream ss;                       \
    ss << __FILE__ << ':' << __LINE__           \
    << " CUDA error: "                       \
    << cudaGetErrorString(status)            \
    << " (" << #call << ")\n";               \
    std::fputs(ss.str().c_str(), stderr);       \
    std::exit(EXIT_FAILURE);                    \
  }                                             \
} while (0)

#endif // CUDAUTILS_H
