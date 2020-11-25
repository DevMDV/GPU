#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
__device__ int  add(int a, int b)
{
  return a + b;
}

__global__ void kernel(int a, int b, int* c)
{
   int q = add(a,b);
   *c = q;
}

int main()
{ int c;
    int *dev_c;
    HANDLE_ERROR (cudaMalloc((void**)&dev_c, sizeof(int)));
    kernel<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR (cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf ("2 + 7 = %d\n", c);
    cudaFree(dev_c);
    return 0;
}

