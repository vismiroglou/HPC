'''
Nested loop implementation of the mandelbrot iterative task. Includes:
-Baseline
-Numba
'''
import numpy as np
from numba import jit, objmode
import time
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

def plot_mandelbrot(img, args):
    # Plots the mandelbrot figure
    if not os.path.isdir('graphics/'):
        os.mkdir('graphics/')

    figure = plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='hot', extent=[args.re_floor, args.re_ceiling, args.im_floor, args.im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.savefig(f"graphics/{args.name}.png")
    plt.show()

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    return re, im

def mandelbrot(re, im, I):
    start = time.time()
    img = np.zeros((len(im), len(re)))
    for r in tqdm(range(len(re))):
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
    from config import config
    config.name = 'loop_mandelbrot'

    re, im = make_grid(config.re_floor, config.re_ceiling, config.im_floor, config.im_ceiling, config.pre, config.pim)

    models = {"Baseline": mandelbrot(re, im, config.I),
              "JIT": mandelbrot_numba(re, im, config.I )}

    for model in models:
        img, proc_time = models[model]
        print(f'Processing time for {model}: {proc_time}[s]')
    
    plot_mandelbrot(img, config)
