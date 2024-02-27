'''
This code implements the loop-version parallelization with multiple different options
to compare performance. The naive loop-version is used as it gets the most out of parallelization.
'''

import numpy as np
import multiprocessing as mp
from naive_mandelbrot import make_grid, mandelbrot, parallelize_grid


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(prog= 'Loop Mandelbrot',
                        description= 'Nested loop implementation for the mandelbrot recursive task. Includes JIT and multiprocessing')
    ap.add_argument('--name', type=str, default='loop_mandelbrot', help='Experiment name')
    ap.add_argument('--pre', type=int, default=5000, help='Amount of real numbers in the grid')
    ap.add_argument('--pim', type=int, default=5000, help='Amount of imaginary numbers in the grid')
    ap.add_argument('--re_floor', type=float, default=-2, help='Lower bound of real component')
    ap.add_argument('--re_ceiling', type=float, default=1, help='Higher bound of real component')
    ap.add_argument('--im_floor', type=float, default=-1.5, help='Lower bound of imaginary component')
    ap.add_argument('--im_ceiling', type=float, default=1.5, help='Higher bound of imaginary component')
    ap.add_argument('--num_iter', type=int, default=100, help='Number of iterations')
    args = ap.parse_args()

    pre = args.pre
    pim = args.pim
    re_floor = args.re_floor
    re_ceiling = args.re_ceiling
    im_floor = args.im_floor
    im_ceiling = args.im_ceiling
    I = args.num_iter

    re, im = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)

    for num_workers in range(12):
        parallelize_grid(re, im, I, num_workers)