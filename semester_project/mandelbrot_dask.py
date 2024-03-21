from mandelbrot_vect import mandelbrot_vector as mandelbrot
from mandelbrot_loop import plot_mandelbrot
import dask.array as da
from dask.distributed import Client, wait
import numpy as np

# import time

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = da.from_array(np.linspace(re_floor, re_ceiling, pre))
    im = da.from_array(np.linspace(im_floor, im_ceiling, pim))
    X, Y = da.from_array(np.meshgrid(re, im))
    c = X + Y * 1j
    return c

def parallelize_grid(config, chunks):
    c = make_grid(config.re_floor, 
                        config.re_ceiling, 
                        config.im_floor, 
                        config.im_ceiling, 
                        config.pre, 
                        config.pim)
    img = c.rechunk(chunks=chunks).map_blocks(lambda x: mandelbrot(x, 100))
    client.close()
    return(img)

if __name__ == '__main__':
    from config import config
    
    num_workers = 6
    client = Client(n_workers=num_workers)
    chunks = (100,100)
    img = parallelize_grid(config, num_workers, chunks)
    img.compute()
    plot_mandelbrot(img, config)
    