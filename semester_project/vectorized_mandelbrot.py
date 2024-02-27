import numpy as np
import multiprocessing as mp
from numba import jit, objmode
from naive_mandelbrot import plot_mandelbrot, plot_time_results
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

# @jit Does not work yet.
# def mandelbrot_vector_numba(c, I):
#     with objmode(start='f8'):
#         start = time.time()
#     z = np.zeros_like(c)
#     img = np.zeros((c.shape[0], c.shape[1]))
#     for i in range(I):
#         mask = np.abs(z) <= 2
#         z[mask] = z[mask] * z[mask] + c[mask]
#         img += mask.astype(int)
#     with objmode(proc_time = 'f8'):
#         proc_time = time.time() - start
#     return img, proc_time

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
    from argparse import ArgumentParser
    ap = ArgumentParser(prog = 'vectorized_mandelbrot.py',
                        description= 'Nested loop implementation for the mandelbrot recursive task. Includes JIT and multiprocessing')
    ap.add_argument('--name', type=str, default='vect_mandelbrot', help='Experiment name')
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

    c = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)

    models = {"Vectorized": mandelbrot_vector(c, I),
            #   "vectorized + JIT": mandelbrot_vector_numba(c, I ),
              "Parallelized": parallel_grid(num_workers, c, I),
            }
    times = []   

    for model in models:
        img, proc_time = models[model]
        print(f'Processing time for {model}: {proc_time}[s]')
        times.append(proc_time)
    
    plot_mandelbrot(img, args)
    plot_time_results(models.keys(), times, args)

  