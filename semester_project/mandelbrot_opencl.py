import pyopencl as cl
import numpy as np
from config import config
import time
from utils import make_grid, plot_mandelbrot

if __name__ == '__main__':
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    grid_host = make_grid(config.re_floor, config.re_ceiling, config.im_floor, config.im_ceiling, config.pre, config.pim, np.float32)
    z_host = np.zeros_like(grid_host)
    result_host = np.zeros_like(grid_host, np.int32)

    mf = cl.mem_flags
    grid_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid_host)
    z_device = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z_host)
    result_device = cl.Buffer(context, mf.WRITE_ONLY, z_host.nbytes)

    kernel_source = open("kernel.cl").read()
    prog = cl.Program(context, kernel_source).build(f"-D rows={z_host.shape[0]} -D cols={z_host.shape[1]}")

    for wgs in [(1,1), (1,1), (5,5), (10,10)]:
        print(wgs)
        start = time.time()
        prog.mandelbrot(queue, z_host.shape, wgs, grid_device, z_device, result_device).wait()
        cl.enqueue_copy(queue, result_host, result_device)
        print(f'{config.pre}x{config.pim} grid-{context.devices}-wgs{wgs}: {time.time()-start}')

    plot_mandelbrot(result_host, config)
    
