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

# gc設定
gc.enable()
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(format=fmt)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

random.seed(0)
np.random.seed(0)


def test_4vars(aegs):
    X1 = generate_sequence(args.x1, args.sample_size)
    E2 = generate_sequence(args.x2, args.sample_size)
    E3 = generate_sequence(args.x3, args.sample_size)
    E4 = generate_sequence(args.x4, args.sample_size)

    f12 = map_randomly(range(args.x1), range(args.x2))
    f13 = map_randomly(range(args.x1), range(args.x3))
    f32 = map_randomly(range(args.x3), range(args.x2))
    f34 = map_randomly(range(args.x3), range(args.x4))

    X3 = [(f13[x1] + e3) % args.x3 for x1, e3 in zip(X1, E3)]
    X2 = [(f12[x1] + f32[x3] + e2) % args.x2 for x1, x3, e2 in zip(X1, X3, E2)]
    X4 = [(f34[x3] + e4) % args.x4 for x3, e4 in zip(X3, E4)]

    data = np.stack([X1, X2, X3, X4], axis=-1)
    estimate, score = ges.fit_nml(data)
#    print(estimate)
    return (estimate == [[0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]]).all()

if  __name__ == "__main__":
    nsim = 100
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--x1", type=int, default=10, help="x1's domain size")
    parser.add_argument("--x2", type=int, default=10, help="x2's domain size")
    parser.add_argument("--x3", type=int, default=10, help="x3's domain size")
    parser.add_argument("--x4", type=int, default=10, help="x4's domain size")
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
    res1 = sum([test_4vars(args) for _ in tqdm(range(nsim))])
    logger.info(f"model1 acc: {res1 / nsim:.3f}")
