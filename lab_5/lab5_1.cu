#include <cstdio>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
constexpr size_t N = 2048; // Количество элементов в массиве.

int main ()
{
  thrust::host_vector<int> v(N, 0);
  for(size_t i=0; i<N; i++)
  {
    v[i] = random() % 1000;
  }
  thrust::device_vector<int> cuda_v = v;

  auto t1 = std::chrono::high_resolution_clock::now();
  thrust::sort(cuda_v.begin(), cuda_v.end());
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() <<std::endl;

  thrust::host_vector<int> result = cuda_v;

  for(size_t i=0; i < 100; i++)
  {
    std::cout << result[i] << " ";
  }

  return 0;
}