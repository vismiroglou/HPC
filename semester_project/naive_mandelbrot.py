'''
Nested loop implementation of the mandelbrot iterative task. Includes:
-Baseline
-Numba
-Parallelized
'''

import numpy as np
from numba import jit, objmode
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
import os

def plot_mandelbrot(img):
    if not os.path.isdir('graphics/'):
        os.mkdir('graphics/')

    figure = plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='hot', extent=[re_floor, re_ceiling, im_floor, im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.savefig("graphics/loop_mandelbrot.png")
    plt.show()

def plot_time_results(algorithms, times, args):
    if not os.path.isdir('graphics/'):
        os.mkdir('graphics/')
    figure = plt.figure(figsize=(5,5))
    plt.plot(algorithms, times)
    plt.xlabel("Algorithm")
    plt.ylabel("Time[s]")
    plt.title(f"Processing time per algorithm on a {args.pre}x{args.pim} grid.")
    plt.savefig("graphics/time_loop_model.png")
    plt.show()

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    return re, im

def parallelize_grid(re, im, I, num_workers):
    start = time.time()
    pool = mp.Pool(num_workers)
    reals = np.array_split(re, num_workers)
    results = [pool.apply_async(mandelbrot, (real, im, I, )) for real in reals]
    pool.close()
    pool.join()
    results = [result.get()[0] for result in results]
    img = np.concatenate(results, axis=1)
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

def mandelbrot(re, im, I):
    start = time.time()
    img = np.zeros((len(im), len(re)))
    for r in range(len(re)):
        for i in range(len(im)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            for iter in range(I):
                if np.abs(z) <= 2:
                    z = z * z + c
                    img[i,r ] += 1
    proc_time = time.time() - start
    return img, proc_time

@jit
def mandelbrot_numba(re, im, I):
    with objmode(start = 'f8'):
        start = time.time()
    img = np.zeros((len(im), len(re)))
    for i in range(len(im)):
        for r in range(len(re)):
            c = re[r] + im[i] * 1j
            z = 0 + 0 * 1j
            for iter in range(I):
                if np.abs(z) <= 2:
                    z = z * z + c
                    img[i, r] += 1
    with objmode(proc_time = 'f8'):
        proc_time = time.time() - start
    return img, proc_time

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

    re, im = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)

    models = {"Baseline": mandelbrot(re, im, I),
              "JIT": mandelbrot_numba(re, im, I ),
              "Parallelized": parallelize_grid(re, im, I, num_workers),
              "Parallelized + JIT": parallelize_grid_numba(re, im, I, num_workers)
            }
    times = []   

    for model in models:
        img, proc_time = models[model]
        print(f'Processing time for {model}: {proc_time}[s]')
        times.append(proc_time)
    
    plot_mandelbrot(img)
    plot_time_results(models.keys(), times, args)
