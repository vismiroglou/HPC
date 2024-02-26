import numpy as np
import multiprocessing as mp
from IPython import get_ipython

def make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim):
    re = np.linspace(re_floor, re_ceiling, pre)
    im = np.linspace(im_floor, im_ceiling, pim)
    X, Y = np.meshgrid(re, im)
    c = X + Y * 1j
    return c

def mandelbrot(c, I):
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
    results = [pool.apply_async(mandelbrot, (subgrid, I, )) for subgrid in subgrids]
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
    I = 100

    ipython = get_ipython()

    c = make_grid(re_floor, re_ceiling, im_floor, im_ceiling, pre, pim)
    
    img = parallel_grid(num_workers, c)

    ipython.run_line_magic("timeit", "parallel_grid(num_workers,c)")

    from matplotlib import pyplot as plt
    figure = plt.figure(figsize=(5,5), dpi=500)
    plt.imshow(img, cmap='hot', extent=[re_floor, re_ceiling, im_floor, im_ceiling])
    plt.xlabel("Re[c]")
    plt.ylabel("Im[c]")
    plt.title("M(c)")
    plt.show()