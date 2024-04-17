from matplotlib import pyplot as plt
import os
import numpy as np

def make_grid(re_floor:float, re_ceiling:float, im_floor:float, im_ceiling:float, pre:int, pim:int, dtype:float) -> np.ndarray:
    """
    Return a pre x pim complex grid c.

    Parameters
    ----------
    re_floor : Float
        Real part lower bound
    re_ceiling : Float
        Real part upper bound
    im_floor : Float
        Imaginary part lower bound
    im_ceiling : Float
        Imaginary part upper bound
    pre : Integer
        Number of points between lower and upper bounds for the real part
    pim : Integer
        Number of points between lower and upper bounds for the imaginary part
    dtype: Data Type
        Data type of real and imaginary parts.


    Returns
    -------
    c : Numpy array of complex numbers
        The pre x pim grid

    # Test case 1: Check output shape
    >>> make_grid(0, 1, 0, 1, 5, 5, float).shape 
    (5, 5)

    # Test case 2: Check output type
    >>> isinstance(make_grid(0, 1, 0, 1, 5, 5, float), np.ndarray)
    True

    # Test case 3: Check output dtype
    >>> make_grid(0, 1, 0, 1, 5, 5, np.float64).dtype
    dtype('complex128')

    """
    re = np.linspace(re_floor, re_ceiling, pre).astype(dtype)
    im = np.linspace(im_floor, im_ceiling, pim).astype(dtype)
    X, Y = np.meshgrid(re, im)
    c = X + Y * 1j
    return c

def mandelbrot_vector(c, I):
    """
    Return a pre x pim numpy array of integers representing the mandelbrot fractal.

    Parameters
    ----------
    c : Numpy array of complex numbers
        The complex grid to perform the mandelbrot operation on.
    I : Integer
        Maximum number of iterations if divergence isn't achieved.

    Returns
    -------
    img : Numpy array of integers
        The mandelbrot fractal  

    # Test case 1: Check if output has correct shape
    >>> c = np.array([[-2 - 2j,  0 - 2j,  2 - 2j], [-2 + 0j,  0 + 0j,  2 + 0j], [-2 + 2j,  0 + 2j,  2 + 2j]])
    >>> I = 100
    >>> mandelbrot_vector(c, I).shape
    (3, 3)

    # Test case 2: Check if output type is correct
    >>> isinstance(mandelbrot_vector(c, I), np.ndarray)
    True

    # Test case 3: Check if output values are within expected range
    >>> img = mandelbrot_vector(c, I)
    >>> np.all(img >= 0) and np.all(img <= I)
    True

    """
    z = np.zeros_like(c)
    img = np.zeros((c.shape[0], c.shape[1]))
    for i in range(I):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] * z[mask] + c[mask]
        img += mask.astype(int)
    return img
    
def plot_mandelbrot(img, args):
    """
    Plots the mandelbrot fractal

    Parameters
    ----------
    img : Numpy array of integers
        The mandelbrot fractal
    args : argparse.Namespace
        Namespace containing the following:
        re_floor : Real part lower bound
        re_ceiling: Real part upper bound
        im_floor: Imaginary part lower bound
        im_ceiling: Imaginary part upper bound

    Returns
    -------
    """
    
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
