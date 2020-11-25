#include <cstdio>
int main (int argc, char * argv [] )
{
    int deviceCount;
    cudaDeviceProp devProp{};
    cudaGetDeviceCount ( &deviceCount );
    printf ( "Found %d devices\n", deviceCount );
    for ( int device = 0; device < deviceCount; device++)
    {cudaGetDeviceProperties ( &devProp, device );
        printf ("Device %d\n", device );
        printf ("Compute capability : %d.%d\n", devProp.major, devProp.minor);
        printf ("Name : %s\n", devProp.name);
        // Полный объем глобальной памяти в Mбайтах:
        printf ("Total Global Mem: %lu\n", (devProp.totalGlobalMem/(1024*1024)));
        printf ("Shared memory per block: %zu\n" , devProp.sharedMemPerBlock );
        printf ("Registers per block : %d\n", devProp.regsPerBlock);
        printf ("Warp size : %d\n", devProp.warpSize);
        printf ("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf ("Total constant memory: %zu\n", devProp.totalConstMem);
        printf ("Clock Rate : %d\n", devProp.clockRate);
        printf ("Texture Alignment : %zu\n", devProp.textureAlignment);
        printf ("Device Overlap : %d\n", devProp.deviceOverlap);
        printf ("Multiprocessor Count: %d\n", devProp.multiProcessorCount);
        printf ("Max Threads Dim : %d %d %d\n", devProp.maxThreadsDim[0],
                devProp.maxThreadsDim[1], devProp.maxThreadsDim[2] );
        printf ("Max Grid Size : %d %d %d\n", devProp.maxGridSize [0],
                devProp.maxGridSize [1], devProp.maxGridSize [2]);
        printf("")
    }
    return 0;
}
