import numpy as np
import json

from imagewect.shape import Shape, OnePixel, Image, UniformDistribution, NormalDistribution, \
    make_square, make_circle, make_favorite_tetris, make_annulus, make_square_annulus, make_clusters, make_swiss_cheese
from imagewect.wect import compute_wect, discretize_wect

def make_distribution(d):
    if d == "uniform":
        return UniformDistribution(65)
    elif d == "n50":
        return NormalDistribution(65, sigma=0.5)
    elif d == "n25":
        return NormalDistribution(65, sigma=0.25)
    elif d == "n17":
        return NormalDistribution(65, sigma=0.17)
    else:
        raise NotImplementedError(f"unknown distribution {d}")

def make_shape(shape):
    if shape == "square":
        return make_square(65)
    elif shape == "circle":
        return make_circle(65)
    elif shape == "tetris":
        return make_favorite_tetris()
    elif shape == "annulus":
        return make_annulus(65)
    elif shape == "square_annulus":
        return make_square_annulus(65)
    elif shape == "clusters":
        return make_clusters(65)
    elif shape == "swiss_cheese":
        return make_swiss_cheese(65)
    else:
        raise NotImplementedError(f"unknown shape {shape}")

def make_images(shape_name, distribution_name):
    d = make_distribution(distribution_name)
    shape = make_shape(shape_name)
    imgs = {}
    for i in range(0,250):
        img = Image(shape, d.make_channel())
        imgs[f"{i}"] = img.img
    np.savez(f"data/images/{shape_name}/{distribution_name}.npz", **imgs)


def compute_wects(shape_name, distribution_name, num_dirs, fe):
    print("computing wects")
    imgs = np.load(f"data/images/{shape_name}/{distribution_name}.npz")
    shape = make_square(65)

    sample_theta = np.expand_dims(np.linspace(0, 2*np.pi, num=num_dirs+1), axis=1)[:-1]
    directions = np.concatenate((np.cos(sample_theta), np.sin(sample_theta)), axis=1)

    hfs = []
    for direction in directions:
        hfs.append(shape.get_height_filtration(direction))
    dws = {}
    ws = {}
    for hf_idx, hf in enumerate(hfs):
        for i in imgs:
            # print(distribution_name, shape_name, i, "dir", hf_idx)
            w = compute_wect(imgs[f"{i}"], hf, fe=fe)
            dw = discretize_wect(w, np.linspace(-45, 45, 91))
            if hf_idx == 0:
                dws[i] = dw
                ws[i] = [w]
            else:
                dws[i] = np.concatenate([dws[i], dw])
                ws[i].append(w)
        print(f"finished {len(imgs)} for {distribution_name} {shape_name} direction {hf_idx}")
    np.savez(f"data/wects_disc/{fe}fe/{num_dirs}dir/{shape_name}/{distribution_name}.npz", **dws)
    jobj = json.dumps(ws)
    with open(f"data/wects_proper/{fe}fe/{num_dirs}dir/{shape_name}/{distribution_name}.json", "w") as f:
        f.write(jobj)


def main():
    shapes = ["square", "annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris"]
    dists = ["n25"]
    dirs = [30]
    for i in dirs:
        for shape in shapes:
            for dist in dists:
                compute_wects(shape, dist, i, "AVG")


main()