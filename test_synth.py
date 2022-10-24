from __future__ import division
import math
import random
import sys
import time
import gc
import logging
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm

sys.path.append("..")
from multicloud import ThreeValCodeLength

# gc設定
gc.enable()
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(format=fmt)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

random.seed(0)
np.random.seed(0)

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f


def generate_sequence(dom_size: int, size: int):
    # generate X from multinomial disctibution
    p_nums = [np.random.random() for _ in range(dom_size)]
    p_vals = [v / sum(p_nums) for v in p_nums]
    X = np.random.choice(a=range(dom_size), p=p_vals, size=size)
    return X
    

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f

def test_model1(args):
    X = generate_sequence(args.x_dom, args.sample_size)
    Y = generate_sequence(args.y_dom, args.sample_size)
    Z = generate_sequence(args.z_dom, args.sample_size)

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 1

def test_model2(args):
    X = generate_sequence(args.x_dom, args.sample_size)
    Y = generate_sequence(args.y_dom, args.sample_size)
    f = map_randomly(range(args.y_dom), range(args.z_dom))
    E_Z = generate_sequence(args.z_dom, args.sample_size)
    Z = [(f2[y] + e_z) % args.z_dom for y, e_z in zip(Y, E_Z)]

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 2

def test_model5(args):
    X = generate_sequence(args.x_dom, args.sample_size)
    f1 = map_randomly(range(args.x_dom), range(args.y_dom))
    f2 = map_randomly(range(args.x_dom), range(args.z_dom))
    E_Y = generate_sequence(args.y_dom, args.sample_size)
    E_Z = generate_sequence(args.z_dom, args.sample_size)
    Y = [(f1[x] + e_y) % args.y_dom for x, e_y in zip(X, E_Y)]
    Z = [(f2[x] + e_z) % args.z_dom for x, e_z in zip(X, E_Z)]

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 5


def test_model8(args):
    E_X = generate_sequence(args.x_dom, args.sample_size)
    Y = generate_sequence(args.y_dom, args.sample_size)
    Z = generate_sequence(args.z_dom, args.sample_size)
    f1 = map_randomly(range(args.y_dom), range(args.x_dom))
    f2 = map_randomly(range(args.z_dom), range(args.x_dom))
    X = [(f1[y] + f2[z] + e_x) % args.x_dom for y, z, e_x in zip(Y, Z, E_X)]

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 8

def test_model11(args):
    X = generate_sequence(args.x_dom, args.sample_size)
    E_Y = generate_sequence(args.y_dom, args.sample_size)
    E_Z = generate_sequence(args.z_dom, args.sample_size)
    f1 = map_randomly(range(args.x_dom), range(args.y_dom))
    f2 = map_randomly(range(args.x_dom), range(args.z_dom))
    f3 = map_randomly(range(args.y_dom), range(args.z_dom))
    Y = [(f1[x] + e_y) % args.y_dom for x, e_y in zip(X, E_Y)]
    Z = [(f2[x] + f3[y] + e_z) % args.z_dom for x, y, e_z in zip(X, Y, E_Z)]

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 11

def test_model17(args):
    E_X = generate_sequence(args.x_dom, args.sample_size)
    E_Y = generate_sequence(args.y_dom, args.sample_size)
    Z = generate_sequence(args.z_dom, args.sample_size)
    f1 = map_randomly(range(args.z_dom), range(args.x_dom))
    f2 = map_randomly(range(args.x_dom), range(args.y_dom))
    X = [(f1[z] + e_x) % args.x_dom for z, e_x in zip(Z, E_X)]
    Y = [(f2[x] + e_y) % args.y_dom for x, e_y in zip(X, E_Y)]

    model = ThreeValCodeLength(X, Y, Z)
    pred = model.predict()
    return pred == 17

if __name__ == "__main__":
    nsim = 100

    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--x_dom", type=int, default=10, help="x domain size")
    parser.add_argument("--y_dom", type=int, default=10, help="y domain size")
    parser.add_argument("--z_dom", type=int, default=10, help="z domain size")
    parser.add_argument("--sample_size", type=int, default=1000, help="sample size")

    args = parser.parse_args()

    res1 = sum([test_model1(args) for _ in tqdm(range(nsim))])
    res2 = sum([test_model2(args) for _ in tqdm(range(nsim))])
    res5 = sum([test_model5(args) for _ in tqdm(range(nsim))])
    res8 = sum([test_model8(args) for _ in tqdm(range(nsim))])
    res11 = sum([test_model11(args) for _ in range(nsim)])
    res17 = sum([test_model17(args) for _ in range(nsim)])

    logger.info(f"mode1 acc: {res1 / nsim:.3f}")
    logger.info(f"mode2 acc: {res2 / nsim:.3f}")
    logger.info(f"mode5 acc: {res5 / nsim:.3f}")
    logger.info(f"mode8 acc: {res8 / nsim:.3f}")
    logger.info(f"mode11 acc: {res11 / nsim:.3f}")
    logger.info(f"mode17 acc: {res17 / nsim:.3f}")

