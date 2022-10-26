from __future__ import division
import os
import sys
import numpy as np
from math import ceil, log, sqrt
from collections import Counter, defaultdict
import itertools
from sklearn.preprocessing import LabelEncoder
import causaldag as cd

from .utils import (
    log2,
    stratify,
    map_to_majority,
    update_regression,
    cause_effect_negloglikelihood
)

class ThreeValCodeLength():
    def __init__(self, X, Y, Z):
        self.X = self.preprocess(X)
        self.Y = self.preprocess(Y)
        self.Z= self.preprocess(Z)

        self.n = len(self.X)
        assert self.n == len(self.Y) == len(self.Z)

        self.X_freqs = Counter(X)
        self.Y_freqs = Counter(Y)
        self.Z_freqs = Counter(Z)

    
    def preprocess(self, X):
        le_X = LabelEncoder()
        transform_X = le_X.fit_transform(X)
        return transform_X
    
    def merge_causes(self, C1, C2):
        C_concat_str =np.core.defchararray.add(C1.astype(str), C2.astype(str))
        le_ = LabelEncoder()
        Cs= le_.fit_transform(C_concat_str)
        return Cs
    
    def exogenousnoise_likelihood(self, freqs):
        loglikelihood = 0
        for freq in freqs.values() :
            loglikelihood += freq * (log2(self.n) - log2(freq))
        return loglikelihood
    
    def effect_likelihood(self, C, E, ):
        loglikelihood = 0
        f = map_to_majority(C, E)
        f = update_regression(C, E, f)
        loglikelihood += cause_effect_negloglikelihood(C, E, f)
        loglikelihood +=  (len(set(C)) - 1) * log2(len(set(E)))
        return loglikelihood
    
    def _model1(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model2(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.Y, self.Z)
        return loglikelihood

    def _model3(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Z, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood

    def _model4(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model5(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Y)
        loglikelihood += self.effect_likelihood(self.X, self.Z)
        return loglikelihood
    
    def _model6(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Y, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.Y, self.Z)
        return loglikelihood

    def _model7(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Z, self.X)
        loglikelihood += self.effect_likelihood(self.Z, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model8(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.Y, self.Z)
        loglikelihood += self.effect_likelihood(Cs, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood

    def _model9(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.Z, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(Cs, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood

    def _model10(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.X, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(Cs, self.Z)
        return loglikelihood
    
    def _model11(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.X, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Y)
        loglikelihood += self.effect_likelihood(Cs, self.Z)
        return loglikelihood
    
    def _model12(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.X, self.Z)
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(Cs, self.Y)
        loglikelihood += self.effect_likelihood(self.X, self.Z)
        return loglikelihood
    
    def _model13(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.Y, self.Z)
        loglikelihood += self.effect_likelihood(Cs, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.Y, self.Z)
        return loglikelihood
    
    def _model14(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.X, self.Y)
        loglikelihood += self.effect_likelihood(self.Y, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(Cs, self.Z)
        return loglikelihood
    
    def _model15(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.X, self.Z)
        loglikelihood += self.effect_likelihood(self.Z, self.X)
        loglikelihood += self.effect_likelihood(Cs, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model16(self):
        loglikelihood = 0
        Cs = self.merge_causes(self.Y, self.Z)
        loglikelihood += self.effect_likelihood(Cs, self.X)
        loglikelihood += self.effect_likelihood(self.Z, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model17(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Z, self.X)
        loglikelihood += self.effect_likelihood(self.X, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def _model18(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Y, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Z)
        return loglikelihood
    
    def _model19(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Y)
        loglikelihood += self.effect_likelihood(self.Y, self.Z)
        return loglikelihood
    
    def _model20(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Y, self.X)
        loglikelihood += self.effect_likelihood(self.Z, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood

    def _model21(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Z, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.Y, self.Z)
        return loglikelihood

    def _model22(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Z)
        loglikelihood += self.effect_likelihood(self.Z, self.Y)
        return loglikelihood
    
    def _model23(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.effect_likelihood(self.Z, self.Y)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood

    def _model24(self):
        loglikelihood = 0
        loglikelihood += self.exogenousnoise_likelihood(self.X_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.effect_likelihood(self.X, self.Z)
        return loglikelihood

    def _model25(self):
        loglikelihood = 0
        loglikelihood += self.effect_likelihood(self.Y, self.X)
        loglikelihood += self.exogenousnoise_likelihood(self.Y_freqs)
        loglikelihood += self.exogenousnoise_likelihood(self.Z_freqs)
        return loglikelihood
    
    def predict(self):
        code_length = [eval(f"self._model{i}()", {"self": self}) for i in range(1, 26)]
        #print(np.argsort(code_length) + 1)
        return np.argmin(code_length) + 1

if __name__ == "__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description="multivariate  experiment")
    parser.add_argument("--N", type=int, default=1000, help="number of samples")
    parser.add_argument("--m0", type=int, default=4, help="number of distinct values of the multinomial r.v X")
    parser.add_argument("--m1", type=int, default=5, help="number of distinct values of the multinomial r.v Y")
    args = parser.parse_args()

    # generate model4: x1 <- x0 -> x2
    rand0 = [np.random.random() for _ in range(args.m0)]
    pvals0 = [rand_f / sum(rand0) for rand_f in rand0]
    rand1 = [np.random.random() for _ in range(args.m1)]
    pvals1 = [rand_f / sum(rand1) for rand_f in rand1]
    x0 = np.random.choice(a=range(args.m0), p=pvals0, size=args.N)
    x1 = (x0 + np.random.choice(a=range(args.m1), p=pvals1, size=args.N)) % args.m1
    x2 = (x0 + np.random.choice(a=range(args.m1), p=pvals1, size=args.N)) % args.m1

    model = ThreeValCodeLength(x0, x1, x2)
    pred = model.predict()
    print(pred)




    
