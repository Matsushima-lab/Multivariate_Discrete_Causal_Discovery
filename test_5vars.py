import numpy as np
import random
import ges
import logging
import gc
from tqdm import tqdm

from test_3vars import (
        map_randomly,
        generate_sequence
        )
from utils import SHD

# gc設定
gc.enable()
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(format=fmt)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

random.seed(0)
np.random.seed(0)


def test_5vars(aegs):
    X1 = generate_sequence(args.x1, args.sample_size)
    X2 = generate_sequence(args.x2, args.sample_size)
    X3 = generate_sequence(args.x3, args.sample_size)
    E4 = generate_sequence(args.x4, args.sample_size)
    E5 = generate_sequence(args.x5, args.sample_size)

    f14 = map_randomly(range(args.x1), range(args.x4))
    f24 = map_randomly(range(args.x2), range(args.x4))
    f34 = map_randomly(range(args.x3), range(args.x4))
    f45 = map_randomly(range(args.x4), range(args.x5))

    #X2 = [(f12[x1] + e2) % args.x2 for x1, e2 in zip(X1, E2)]
    X4 = [(f14[x1] + f24[x2] + f34[x3] + e4) % args.x4 for x1, x2, x3, e4 in zip(X1, X2, X3, E4)]
    X5 = [(f45[x4] + e5) % args.x5 for x4, e5 in zip(X4, E5)]

    data = np.stack([X1, X2, X3, X4, X5], axis=-1)
    estimate, score = ges.fit_nml(data, phases = ['forward'])
    #print(estimate)
    true_graph=  [[0, 0, 0, 1, 0], 
        [0, 0, 0, 1, 0], 
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]
    return (estimate == true_graph).all(), SHD(estimate, true_graph, double_for_anticausal=False)

if  __name__ == "__main__":
    nsim = 100
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--x1", type=int, default=10, help="x1's domain size")
    parser.add_argument("--x2", type=int, default=10, help="x2's domain size")
    parser.add_argument("--x3", type=int, default=10, help="x3's domain size")
    parser.add_argument("--x4", type=int, default=10, help="x4's domain size")
    parser.add_argument("--x5", type=int, default=10, help="x5's domain size")
    parser.add_argument("--sample_size", type=int, default=1000, help="sample size")
    args = parser.parse_args()


    #from ges.scores.nml_pen import NmlPen
    #model = NmlPen(data)
    #print(model.full_score(A=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])))
    #print(model.full_score(A=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])))
    #print(model.full_score(A=np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])))
    #print(model.full_score(A=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])))
    #print(model.full_score(A=np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])))
    #print(model.full_score(A=np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])))
    acc_res= 0
    shd_res = [None] * nsim
    for i in tqdm(range(nsim)):
        correct, shd = test_5vars(args)
        acc_res += correct
        shd_res[i] = shd
    logger.info(f"5vars SHD result: {np.mean(shd_res):.3f} ± {np.std(shd_res):.3f}")
    logger.info(f"5vars acc result: {acc_res / nsim:.3f}")
