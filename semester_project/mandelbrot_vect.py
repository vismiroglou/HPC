import numpy as np
import multiprocessing as mp
from numba import jit, objmode
from mandelbrot_loop import plot_mandelbrot
import time

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    X, Y = np.meshgrid(re, im)
    c = X + Y * 1j
    return c

def mandelbrot_vector(c, I):
    start = time.time()
    z = np.zeros_like(c)
    img = np.zeros((c.shape[0], c.shape[1]))
    for i in range(I):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] * z[mask] + c[mask]
        img += mask.astype(int)
    proc_time = time.time() - start
    return img, proc_time

@jit
def mandelbrot_vector_numba(c, I):
    with objmode(start='f8'):
        start = time.time()
    z = np.zeros_like(c)
    orig_shape = c.shape
    c = c.flatten()
    z = z.flatten()
    img = np.zeros(c.shape)
    for i in range(I):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] * z[mask] + c[mask]
        img += mask.astype(np.int_)
    with objmode(proc_time = 'f8'):
        proc_time = time.time() - start
    img = img.reshape(orig_shape)
    return img, proc_time

def parallel_grid(num_workers, c, I):
    start = time.time()
    subgrids = np.array_split(c, num_workers)
    pool = mp.Pool(num_workers)
    results = [pool.apply_async(mandelbrot_vector, (subgrid, I, )) for subgrid in subgrids]
    pool.close()
    pool.join()
    results = [result.get()[0] for result in results]
    img = np.concatenate(results)
    proc_time = time.time() - start
    return img, proc_time

if __name__ == '__main__':
    from config import config
    config.name = 'vect_mandelbrot'

    c = make_grid(config.re_floor, config.re_ceiling, config.im_floor, config.im_ceiling, config.pre, config.pim)

    models = {"Vect": mandelbrot_vector(c, config.I),
              "Vect + JIT": mandelbrot_vector_numba(c, config.I)}
    
    for model in models:
        img, proc_time = models[model]
        print(f'Processing time for {model}: {proc_time}[s]')

    plot_mandelbrot(img, config)

  