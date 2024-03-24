from mandelbrot_vect import make_grid
import dask.array as da
from dask.distributed import Client, wait
import numpy as np
import time
from matplotlib import pyplot as plt

def mandelbrot(c, I):
    z = np.zeros_like(c)
    img = np.zeros((c.shape[0], c.shape[1]))
    for i in range(I):
        mask = np.abs(z) <= 2
        # Early stopping optimization
        if not mask.any():
            break
        z[mask] = z[mask] * z[mask] + c[mask]
        img += mask.astype(int)
    return img

if __name__ == '__main__':
    from config import config
    
    # Dask automatically parallelizes processes when creating a Client.
    # Changing between local and Strato just depends on setting up the Client properly.
    num_workers = 6
    I = 100

    client = Client(n_workers=num_workers)
    # client = Client('10.92.1.192:8786')

    # Creating the grid as a numpy array
    c = make_grid(config.re_floor, 
                        config.re_ceiling, 
                        config.im_floor, 
                        config.im_ceiling, 
                        config.pre, 
                        config.pim,
                        np.float32)
    
    points = [[],[]]

    # Simple numpy implementation
    print(f'Starting numpy version')
    start = time.time()
    img = mandelbrot(c, I)
    proc_time = time.time() - start
    print(f'Processing time{proc_time}')
    points[0].append('numpy')
    points[1].append(proc_time)

    # Chunk size comparison 1-D
    print(f'Starting dask distributed version for multiple chunk sizes:')
    
    c = da.from_array(c)
    chunk_sizes_1 = [(config.pre//x, config.pim) for x in [2, 3, 6, 9, 12, 15, 18]]
    chunk_sizes_2 = [(1000, 1000), (800,800), (600,600), (400,400), (200,200), (100,100)]
    chunk_sizes_3 = [(200, 200), (100,100), (80,80), (50,50), (20,20), (10,10)]
    for chunk_size in chunk_sizes_3:
        chunks = chunk_size
        print(f'Chunk size: {chunks}')
        start = time.time()
        img = c.rechunk(chunks=chunks).map_blocks(lambda x: mandelbrot(x, I)).compute()
        wait(img)
        proc_time = time.time() - start
        print(f'Processing time{proc_time}')
        points[0].append(str(chunks))
        points[1].append(proc_time)

    client.close()
    fig = plt.figure(figsize=(7,5))
    plt.xticks(rotation=30, fontsize=8, ha='right')
    plt.ylabel('time[s]')
    plt.title(config.title)
    plt.plot(points[0], points[1])
    plt.savefig(f'handin_2/{config.name}')
    plt.show()

    

    
    
