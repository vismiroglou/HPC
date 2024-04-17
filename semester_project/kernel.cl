#include <pyopencl-complex.h>
__kernel void mandelbrot(
    __global const cfloat_t *grid_device,
    __global       int      *result_device)
{   
    cfloat_t z_device = cfloat_new(0,0);
    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = i * cols + j ;

    cfloat_t result = cfloat_new(0,0);

    for (int k = 0; k < 100; ++k)
    {
        if (cfloat_abs(z_device) < 2) 
        {
            z_device = cfloat_add(grid_device[index], cfloat_mul(z_device, z_device));
            result_device[index] += 1;
        }
    }
}