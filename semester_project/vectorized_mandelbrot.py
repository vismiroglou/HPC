import numpy as np
import multiprocessing as mp
from IPython import get_ipython
from numba import jit

def plot_mandelbrot(img):
    from matplotlib import pyplot as plt
    figure = plt.figure(figsize=(5,5), dpi=500)
    plt.imshow(img, cmap='hot', extent=[re_floor, re_ceiling, im_floor, im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.show()

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    X, Y = np.meshgrid(re, im)
    c = X + Y * 1j
    return c

def mandelbrot_naive(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim, I):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    img = np.zeros((pre, pim))
    for r in range(len(re)):
        for i in range(len(im)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            iter = 0
            while (iter < I) and (np.abs(z) <= 2):
                z = z * z + c
                iter += 1
            img[r, i] = iter
    return(img)

@jit
def mandelbrot_naive_numba(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim, I):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    img = np.zeros((pre, pim))
    for r in range(len(re)):
        for i in range(len(im)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            iter = 0
            while (iter < I) and (np.abs(z) <= 2):
                z = z * z + c
                iter += 1
            img[r, i] = iter
    return(img)

def parallel_naive(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim, I, num_workers):
    
    pool = mp.Pool(num_workers)
    results = [pool.apply_async(mandelbrot_vector, (subgrid, I, )) for subgrid in subgrids]


def mandelbrot_vector(c, I):
    z = np.zeros_like(c)
    img = np.zeros((c.shape[0], c.shape[1]))
    for i in range(I):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] * z[mask] + c[mask]
        img += mask.astype(int)
    return(img)

def parallel_grid(num_workers, c):
    subgrids = np.array_split(c, num_workers)
    pool = mp.Pool(num_workers)
    results = [pool.apply_async(mandelbrot_vector, (subgrid, I, )) for subgrid in subgrids]
    pool.close()
    pool.join()
    results = [result.get() for result in results]
    img = np.concatenate(results)
    return img

if __name__ == '__main__':

    pre = 5000
    pim = 5000
    re_floor = -2
    re_ceiling = 1
    im_floor = -1.5
    im_ceiling = 1.5
    num_workers = 6
    I = 5

    ipython = get_ipython()
    
    ipython.run_line_magic("timeit", "mandelbrot_naive(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim, I)")
    ipython.run_line_magic("timeit", "mandelbrot_naive_numba(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim, I)")

    c = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)
    ipython.run_line_magic("timeit", "mandelbrot_vector(c, I)")
    ipython.run_line_magic("timeit", "parallel_grid(num_workers,c)")

  