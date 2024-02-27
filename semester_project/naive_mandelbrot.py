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
    if not os.path.isdir('graphics/'):
        os.mkdir('graphics/')

    figure = plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='hot', extent=[re_floor, re_ceiling, im_floor, im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.savefig("graphics/mandelbrot.png")
    plt.show()

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    return re, im

def parallelize_grid(re, im, I, num_workers):
    pool = mp.Pool(num_workers)
    reals = np.array_split(re, num_workers)
    imaginaries = np.array_split(im, num_workers)
    results = [pool.apply_async(mandelbrot, (real, im, I, )) for real in reals]
    pool.close()
    pool.join()
    results = [result.get() for result in results]
    result = np.concatenate(results, axis=1)
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
                    img[i,r ] += 1
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
    from argparse import ArgumentParser
    ap = ArgumentParser(prog= 'Loop Mandelbrot',
                        description= 'Nested loop implementation for the mandelbrot recursive task. Includes JIT and multiprocessing')
    ap.add_argument('--pre', type=int, default=5000, help='Amount of real numbers in the grid')
    ap.add_argument('--pim', type=int, default=5000, help='Amount of imaginary numbers in the grid')
    ap.add_argument('--re_floor', type=float, default=-2, help='Lower bound of real component')
    ap.add_argument('--re_ceiling', type=float, default=1, help='Higher bound of real component')
    ap.add_argument('--im_floor', type=float, default=-1.5, help='Lower bound of imaginary component')
    ap.add_argument('--im_ceiling', type=float, default=1.5, help='Higher bound of imaginary component')
    ap.add_argument('--num_iter', type=int, default=100, help='Number of iterations')
    ap.add_argument('--num_workers', type=int, default=6, help='Number of parallel processes to run / number of cores')
    args = ap.parse_args()

    pre = args.pre
    pim = args.pim
    re_floor = args.re_floor
    re_ceiling = args.re_ceiling
    im_floor = args.im_floor
    im_ceiling = args.im_ceiling
    num_workers = args.num_workers
    I = args.num_iter

    ipython = get_ipython()

    re, im = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)
    img = mandelbrot_numba(re, im, I)
    plot_mandelbrot(img)

    # print('Processing time for baseline model:')
    # ipython.run_line_magic("timeit", "mandelbrot(re, im, I)")
    print('Processing time for just-in-time compilation:')
    ipython.run_line_magic("timeit", "mandelbrot_numba(re, im, I)")
    print('Processing time for the parallelized baseline:')
    ipython.run_line_magic("timeit", "parallelize_grid(re, im, I, num_workers)")
    print('Processing time for parallelized with just-in-time:')
    ipython.run_line_magic("timeit", "parallelize_grid_numba(re, im, I, num_workers)")