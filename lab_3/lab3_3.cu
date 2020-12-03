#include "cstdio"
#include <iostream>
#include <chrono>

constexpr size_t SIZE = 16384 * 3; // 16384 * 3
constexpr size_t BLOCK_COUNT = 4096; //for shared alg 16384 * 3 for simple
constexpr size_t BLOCK_SIZE = SIZE / BLOCK_COUNT;
constexpr size_t THREAD_PER_BLOCK = 128;


/*
 * GPU Elapsed time 23.9948
 * CPU Elapsed time 6720
 */

template<typename T>
__global__ void sumMatrixRowShared(T* matrix, T* result)
{
  __shared__ float data[THREAD_PER_BLOCK];
  for(size_t row_num = BLOCK_SIZE * blockIdx.x;
      row_num < BLOCK_SIZE * (blockIdx.x + 1); row_num++) {
    size_t row_start = row_num * SIZE;
    size_t idx = threadIdx.x;
    data[idx] = matrix[row_start + idx];
    for (size_t i=1; i * THREAD_PER_BLOCK + idx < SIZE; i++)
    {
      data[idx] = data[idx] + matrix[row_start + i * THREAD_PER_BLOCK + idx];
    }
    __syncthreads();
    for (size_t s = 1; s < blockDim.x; s <<= 1 )
    {
      size_t index = 2 * s * idx;
      if ( index < blockDim.x )
        data [index] += data [index + s];
      __syncthreads ();
    }

    if (idx == 0)
      result[row_num] = data[0];
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
  auto* result_1 = new float[SIZE];
  //Инициализируем значения векторов
  for (int i = 0; i < SIZE * SIZE; i++)
  {
    matrix[i] = int(i/SIZE);
    result[i%SIZE] = 0;
    result_1[i%SIZE] = 0;
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

  std::cout << "GPU Elapsed time " << time << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  sumMatrixRowCPU(matrix, result_1);
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "CPU Elapsed time " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() <<std::endl;

  for (int i = 0; i < 10; i++)
  {
    printf("Element #%i: %.1f %1.f\n", i , result[i], result_1[i]);
  }

  // Освобождаем ресурсы
  cudaEventDestroy(syncEvent);
  cudaFree(gpu_matrix);
  cudaFree(gpu_result);
  delete[] result;
  delete[] matrix;
}
