import numpy as np
from IPython import get_ipython
from numba import jit

ipython = get_ipython()
@jit
def basicloop(a, b):
    i = 0
    s = 0
    while i < len(b) :
        s += a[i]*b[i]
        i += 1
    return s

@jit
def unroll(a, b):
    i = 0
    s1 = 0
    s2 = 0
    while i < int(len(a)/2-1):
        s1 += a[2*i] * b[2*i]
        s2 += a[2*i+1]*b[2*i+1]
        i += 1
    return s1+s2

@jit
def unroll4(a, b):
    i = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    while i < int(len(a)/4-1):
        s1 += a[4*i] * b[4*i]
        s2 += a[4*i+1]*b[4*i+1]
        s3 += a[4*i+2]*b[4*i+2]
        s4 += a[4*i+3]*b[4*i+3]
        i += 1
    return s1+s2+s3+s4

@jit
def unroll8(a, b):
    i = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = 0
    s8 = 0
    while i < int(len(a)/8-1):
        s1 += a[8*i] * b[8*i]
        s2 += a[8*i+1]*b[8*i+1]
        s3 += a[8*i+2]*b[8*i+2]
        s4 += a[8*i+3]*b[8*i+3]
        s5 += a[8*i+4]*b[8*i+4]
        s6 += a[8*i+5]*b[8*i+5]
        s7 += a[8*i+6]*b[8*i+6]
        s8 += a[8*i+7]*b[8*i+7]
        i += 1
    return s1+s2+s3+s4+s5+s6+s7+s8

@jit
def numdot(a, b):
    s = np.dot(a,b)
    return s

N = 10**6
a = np.random.rand(N)
b = np.random.rand(N)

ipython.run_line_magic("timeit", "basicloop(a,b)")
ipython.run_line_magic("timeit", "unroll(a,b)")
ipython.run_line_magic("timeit", "unroll4(a,b)")
ipython.run_line_magic("timeit", "unroll8(a,b)")
ipython.run_line_magic("timeit", "numdot(a,b)")