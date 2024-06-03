""" Script to analyze structural heterogeneity in chromatin traces."""

import sys
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from pathlib import Path
from p_tqdm import p_map
from multiprocessing import Pool
from itertools import combinations


def Q_factor_parallel(zxys, savepath, string, delta=100.0, **kwargs):
    """ Localization error is around 100nm"""
    conf_pairs = list(combinations(zxys, 2))
    id_pairs = list(combinations(np.arange(0, len(zxys)), 2))
    id_pairs = np.array(id_pairs)

    def compute_Qij(pair):
        pos_i, pos_j = pair
        dist_i = pdist(pos_i, metric="euclidean")
        dist_j = pdist(pos_j, metric="euclidean")
        #only consider pairs of loci that are not NaNs
        disti = dist_i[~np.isnan(dist_i) & ~np.isnan(dist_j)]
        distj = dist_j[~np.isnan(dist_i) & ~np.isnan(dist_j)]
        Qij = np.exp(-((disti - distj)**2 / (2*delta**2))).mean()
        return Qij

    Qlist = p_map(compute_Qij, conf_pairs, **kwargs)
    df = pd.DataFrame()
    df["i"] = id_pairs[:,0]
    df["j"] = id_pairs[:, 1]
    df["Q"] = Qlist
    df.to_csv(Path(savepath) / f"Q_{string}.csv", index=False)


if __name__ == "__main__":
    with open('zxys_filtered_chr2.pkl', 'rb') as f:
        zxys_filtered = pickle.load(f)
    print(len(zxys_filtered))
    print(zxys_filtered[0].shape)
    Q_factor_parallel(zxys_filtered, "data", "chr2", delta=125.0) 
