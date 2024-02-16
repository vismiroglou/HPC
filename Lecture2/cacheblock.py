from IPython import get_ipython
import numpy as np
from numba import jit
ipython = get_ipython()

l1size = int(64*1024/8)

@jit
def baseline(NRUNS, size):
    i = 0
    array = np.random.random(size)
    while i < NRUNS:
        j = 0
        while j < size:
            array[j] = 2.3*array[j]+1.2
            j += 1
        i += 1

def cache_block_l1(NRUNS, size):
    i = 0
    array = np.random.random(size)
    while i < NRUNS:
        j = 0
    while j < size:
        array[j] = 2.3*array[j]+1.2
        j += 1
        i += 1

if __name__ == '__main__':

    NRUNS = 5000
    size = 10
    
    ipython.run_line_magic("timeit", "array2 = baseline(NRUNS, size)")
# def cacheblock():
# for (b=0; b<size/l1size; b++) {
# blockstart = 0;
# for (i=0; i<NRUNS; i++) {
# for (j=0; j<l1size; j++)
# array[blockstart+j] = 2.3*array[blockstart+j]+1.2;
# }
# blockstart += l1size;
# }