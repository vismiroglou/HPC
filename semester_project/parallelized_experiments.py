'''
This code implements the loop-version parallelization with multiple different options
to compare performance. The naive loop-version is used as it gets the most out of parallelization.
'''

import numpy as np
import multiprocessing as mp
from mandelbrot_loop import make_grid, mandelbrot, plot_mandelbrot
from mandelbrot_vect import make_grid as make_grid_vect, mandelbrot_vector
import time

def parallelize_grid(config, num_workers, model):
    start = time.time()
    pool = mp.Pool(num_workers)
    if model == 'loop':
        re, im = make_grid(config.re_floor, 
                           config.re_ceiling, 
                           config.im_floor, 
                           config.im_ceiling, 
                           config.pre, 
                           config.pim)
        reals = np.array_split(re, num_workers)
        results = [pool.apply_async(mandelbrot, (real, im, config.I, )) for real in reals]
        pool.close()
        pool.join()
        results = [result.get()[0] for result in results]
        img = np.concatenate(results, axis=1)
    elif model == 'vect':
        c = make_grid_vect(config.re_floor,
                      config.re_ceiling,
                      config.im_floor,
                      config.im_ceiling,
                      config.pre,
                      config.pim)
        subgrids = np.array_split(c, num_workers)
        results = [pool.apply_async(mandelbrot_vector, (subgrid, config.I, )) for subgrid in subgrids]
        pool.close()
        pool.join()
        results = [result.get()[0] for result in results]
        img = np.concatenate(results)
    else:
        raise ValueError('Invalid model type.')
    proc_time = time.time() - start
    return img, proc_time

def parallelize_grid_numba(re, im, I, num_workers):
    start = time.time()
    pool = mp.Pool(num_workers)
    reals = np.array_split(re, num_workers)
    results = [pool.apply_async(mandelbrot_numba, (real, im, I, )) for real in reals]
    pool.close()
    pool.join()
    results = [result.get()[0] for result in results]
    img = np.concatenate(results, axis=1)
    proc_time = time.time()-start
    return img, proc_time

if __name__ == '__main__':
    from config import config
    config.name = 'parallelize_experiments'

    num_workers = 6
    img, proc_time = parallelize_grid(config, num_workers, 'loop')
    plot_mandelbrot(img, config)

    #Checking the performance of different number of workers
    times = []
    for num_workers in range(1, 13):
        _, proc_time = parallelize_grid(config, num_workers, 'loop')
        times.append(proc_time)

    from matplotlib import pyplot as plt
    plt.plot(range(1, 13), times)
    plt.xlabel('Num_workers')
    plt.ylabel('Proc_time')
    plt.show()


    