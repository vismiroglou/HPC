'''
Nested loop implementation of the mandelbrot iterative task. Includes:
-Baseline
-Numba
-Parallelized
'''

import numpy as np
from numba import jit
import multiprocessing as mp
from IPython import get_ipython

def plot_mandelbrot(img):
    from matplotlib import pyplot as plt
    import os
    if not os.path.isdir('figures/'):
        os.mkdir('figures/')

    figure = plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='hot', extent=[re_floor, re_ceiling, im_floor, im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.savefig("figures/mandelbrot.png")
    plt.show()

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    return re, im

def parallelize_grid(re, im, I, num_workers):
    pool = mp.Pool(num_workers)
    reals = np.array_split(re, num_workers)
    imaginaries = np.array_split(im, num_workers)
    results = [pool.apply_async(mandelbrot, (real, imaginary, I, )) for real, imaginary in zip(reals, imaginaries)]
    pool.close()
    pool.join()
    results = [result.get() for result in results]
    result = np.concatenate(results)
    return result

def parallelize_grid_numba(re, im, I, num_workers):
    pool = mp.Pool(num_workers)
    reals = np.array_split(re, num_workers)
    results = [pool.apply_async(mandelbrot_numba, (real, im, I, )) for real in reals]
    pool.close()
    pool.join()
    results = [result.get() for result in results]
    img = np.concatenate(results, axis=1)
    return img

def mandelbrot(re, im, I):
    img = np.zeros((len(im), len(re)))
    for r in range(len(re)):
        for i in range(len(im)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            for iter in range(I):
                if np.abs(z) <= 2:
                    z = z * z + c
                    img[r, i] += 1
    return(img)

@jit
def mandelbrot_numba(re, im, I):
    img = np.zeros((len(im), len(re)))
    for i in range(len(im)):
        for r in range(len(re)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            for iter in range(I):
                if np.abs(z) <= 2:
                    z = z * z + c
                    img[i, r] += 1
    return(img)

if __name__ == '__main__':
    pre = 5000
    pim = 5000
    re_floor = -2
    re_ceiling = 1
    im_floor = -1.5
    im_ceiling = 1.5
    num_workers = 6
    I = 100

    ipython = get_ipython()

    re, im = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)
    img = mandelbrot_numba(re, im, I)
    plot_mandelbrot(img)

    ipython.run_line_magic("timeit", "mandelbrot(re, im, I)")
    ipython.run_line_magic("timeit", "mandelbrot_numba(re, im, I)")
    ipython.run_line_magic("timeit", "parallelize_grid(re, im, I, num_workers)")
    ipython.run_line_magic("timeit", "parallelize_grid_numba(re, im, I, num_workers)")