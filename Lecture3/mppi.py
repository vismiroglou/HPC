'''
Monte Carlo approximation of pi using multiprocessing.
Random points in (0,1) are generated and pi is calculated from the formula P = pi/4
where P is the ratio of points inside the unitary circle vs. the ones outside.

Multiprocessing tests happen in two manners:
In measure, the processing time is measured as a function of parallelized processes with a maximum
equal to the available cpu cores.

In measure N, the processing time is measured as a function of realizations, with the mp processes
set to exactly the maximum available cpu cores. The number of realizations start from 1 and go higher
than the available processes.
'''

import multiprocessing as mp
import random
import numpy as np
import time
from matplotlib import pyplot as plt

def icircle(L):
    M = 0
    for _ in range(L):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1:
            M += 1
    return(M)

def parpi(num_workers, L, N):
    #Create number of workers
    pool = mp.Pool(num_workers)
    # Get the result as a list of mp.async_results
    results = [pool.apply_async(icircle, (L,)) for i in range(N)]
    # Close and join the processors
    pool.close()
    pool.join()
    # Sum the outputs of all realizations to get the total number of points within the circle
    points_in_circle = np.sum([result.get() for result in results])
    # Calculate pi approximation as P = pi/4 where P is the ratio of points in the circle over points outside.
    mcpi = 4 * points_in_circle / (N*L)
    return(mcpi) 

def measure(L, N, familyname = 'num_workers'):
    '''
    Measures the speed of the parallel processing as a function of the number of available processes.
    The available processes start for 1 (no parallel processing) and go up to maximum available cores.
    '''
    times = []
    for num_workers in range(1, mp.cpu_count()+1):
        start = time.time()
        parpi(num_workers, L, N)
        times.append(time.time()-start)

    #Plot
    plt.plot(range(1, mp.cpu_count()+1), times)
    plt.xticks(range(1, mp.cpu_count()+1))
    plt.xlabel('Number of workers')
    plt.ylabel('Time (s)')
    plt.title('Processing time as a function of parallel processes with the maximum being equal to the number of cores')
    plt.savefig(f'plots/{familyname}_speedup.pdf')
    plt.show()

def measureN(k, Nmax, familyname='num_realizations'):
    '''
    With the number of workers equal to the available pc cores:
    Measures the processing time as a function of realizations while maintaining the total number
    of points N*L constant.
    '''
    times = []
    for N in range(1, Nmax +1):
        L = int(k/N)
        start = time.time()
        parpi(mp.cpu_count(), L, N)
        times.append(time.time()-start)
    
    #Plotting
    plt.plot(range(1, Nmax+1), times)
    plt.xlabel('N: Number of realizations')
    plt.ylabel('Processing time in s')
    plt.title('Processing time as a function of N while keeping the total number of points static (k=N*L). Number of cores: 6')
    plt.xticks(range(1, Nmax +1))
    plt.xlim(1, Nmax+1)
    plt.savefig(f'plots/{familyname}_fixedk.pdf')
    plt.show()
    plt.close()
    

if __name__ == '__main__':
    L = 10**6
    N = 18
    measure(L, N)
    measureN(L*N, 18)