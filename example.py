import numpy as np
import pandas as pd

from pywect.datasets.shape_maker import make_square, make_circle, make_random_tetris, \
                                        make_favorite_tetris, make_clusters, make_swiss_cheese, \
                                        make_annulus
from pywect.wect import get_multidirectional_wects

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def show_img(img):
    plt.imshow(img, vmin=0, vmax=1)
    plt.colorbar()
    plt.axis("off")
    plt.show()

def show_wects(wects):
    fig, ax = plt.subplots()
    for i in range(0,wects.shape[0]):
        ax.plot(wects[i], label=f"{i+1}")
        ax.legend(title="Directions")
        ax.set(ylim=(-25, 2))
    plt.show()

def main():

    img = make_square(64, "normal", mu=0.5, sigma=0.25)
    show_img(img)

    wects = get_multidirectional_wects(img)
    show_wects(wects)


main()