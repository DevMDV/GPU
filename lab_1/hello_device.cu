#include <cstdio>

__global__ void kernel()
{
}

int main()
{
    kernel<<<1, 1>>>();
    printf ("Hello, CUDA!\n");
    return 0;
}