#include "cstdio"
#include <iostream>
#include <chrono>
#include "cuda.h"
#include <cuda_runtime_api.h>

constexpr size_t SIZE = 16384 * 3; // 16384 * 3
constexpr size_t BLOCK_COUNT = 1024 + 512; //for shared alg 16384 * 3 for simple
constexpr size_t THREAD_PER_BLOCK = SIZE / BLOCK_COUNT;
constexpr size_t BLOCK_SIZE = SIZE / BLOCK_COUNT;
constexpr size_t SHARED_MEMORY = 16384 * 3; // opt 16384 * 3 for shared alg
constexpr size_t SHARED_MEMORY_COUNT = SHARED_MEMORY / sizeof(float);

/*
 * GPU Elapsed time 409.04
 * CPU Elapsed time 6828
*/


template<typename T>
__global__ void sumMatrixRowShared(const float* matrix, T* result)
{
  __shared__ float ss[SHARED_MEMORY / sizeof(T)];
  unsigned int idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;
  idx = idx + (SIZE/BLOCK_COUNT) * block_idx;
  result[idx] = 0;

  constexpr int SHARED_SIZE = SHARED_MEMORY / sizeof(T) / THREAD_PER_BLOCK;
  for (size_t batch_num=0; batch_num < SIZE / SHARED_SIZE; batch_num++)
  {
    for (size_t i=0; i < SHARED_SIZE; i++)
    {
      ss[SHARED_SIZE * threadIdx.x + i] = matrix[idx * SIZE + batch_num * SHARED_SIZE + i];
    }

    for(size_t i=0; i < SHARED_SIZE; i++)
    {
      result[idx] += ss[SHARED_SIZE * threadIdx.x + i];
    }
  }
}


template <typename T>
void sumMatrixRowCPU(const float* matrix, T* result)
{
  for(int idx = 0; idx < SIZE; idx++)
  {
    result[idx] = 0;
    for(size_t i=0; i < SIZE; i++)
    {
      result[idx] = result[idx] + matrix[idx * SIZE + i];
    }
  }
}

__host__ int main()
{
  //Выделяем память под вектора
  auto* matrix = new float[SIZE * SIZE];
  auto* result = new float[SIZE];
  //Инициализируем значения векторов
  for (int i = 0; i < SIZE * SIZE; i++)
  {
    matrix[i] = int(i/SIZE);
    result[i%SIZE] = 0;
  }

  float* gpu_matrix;
  float* gpu_result;
  //Выделяем память для векторов на видеокарте
  cudaMalloc((void**)&gpu_matrix, sizeof(float) * SIZE * SIZE);
  cudaMalloc((void**)&gpu_result, sizeof(float) * SIZE);
  cudaMemcpy(gpu_matrix, matrix, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_result, result, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

  dim3 gridSize = dim3(BLOCK_COUNT, 1, 1); //Размер используемой сетки
  dim3 blockSize = dim3(THREAD_PER_BLOCK, 1, 1); //Размер используемого блока
  //Выполняем вызов функции ядра

  cudaEvent_t kernel_start;
  cudaEventCreate(&kernel_start);
  cudaEventRecord(kernel_start, 0);

  sumMatrixRowShared<<<gridSize, blockSize>>>(gpu_matrix, gpu_result);

  cudaEvent_t syncEvent; //Дескриптор события
  cudaEventCreate(&syncEvent); //Создаем event
  cudaEventRecord(syncEvent, 0); //Записываем event
  cudaEventSynchronize(syncEvent); //Синхронизируем event
  float time;
  cudaEventElapsedTime(&time, kernel_start, syncEvent);

  cudaMemcpy(result, gpu_result, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

  //Результаты расчета
  for (int i = 0; i < 10; i++)
  {
    printf("Element #%i: %.1f\n", i , result[i]);
  }
  std::cout << "GPU Elapsed time " << time << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  sumMatrixRowCPU(matrix, result);
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "CPU Elapsed time " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() <<std::endl;

  for (int i = 0; i < 10; i++)
  {
    printf("Element #%i: %.1f\n", i , result[i]);
  }

  // Освобождаем ресурсы
  cudaEventDestroy(syncEvent);
  cudaFree(gpu_matrix);
  cudaFree(gpu_result);
  delete[] result;
  delete[] matrix;
}
