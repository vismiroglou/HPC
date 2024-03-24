from matplotlib import pyplot as plt
import os

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