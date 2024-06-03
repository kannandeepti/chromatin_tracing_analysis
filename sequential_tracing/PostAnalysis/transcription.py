""" Script to analyze genome structure around TSSs."""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pickle
from p_tqdm import p_map
from tqdm import tqdm

def correlator(zxy):
    #note diagonal elements of this are zero!
    rirj = squareform(pdist(zxy, lambda u, v: np.dot(u,v)))
    rsquared = np.sum(zxy * zxy, axis=1)
    #add r^2 to diagonal of matrix returned by squareform
    rirj[np.diag_indices(rirj.shape[0])] = rsquared
    return rirj

def distmap(zxy):
    #standard euclidean distances between coordinates
    return squareform(pdist(zxy))

def compute_rirj_combined_list():
    with open('data_combined_chr21.pkl', 'rb') as f:
        data_combined = pickle.load(f)

    rirjmap_combined_list = p_map(correlator, data_combined['dna_zxys'])
    with open('rirj_combined_list.pkl', 'wb') as f:
        pickle.dump(rirjmap_combined_list, f)

def compute_distmap_combined_list():
    with open('data_combined_chr21.pkl', 'rb') as f:
        data_combined = pickle.load(f)

    distmap_combined_list = p_map(distmap, data_combined['dna_zxys'])
    with open('distmap_combined_list.pkl', 'wb') as f:
        pickle.dump(distmap_combined_list, f)

def compute_gene_pileups():
    with open('data_combined_chr21.pkl', 'rb') as f:
        data_combined = pickle.load(f)
    with open('rirj_combined_list.pkl', 'rb') as f:
        rirjmap_combined_list = pickle.load(f)
    with open('distmap_combined_list.pkl', 'rb') as f:
        distmap_combined_list = pickle.load(f)
    # normalize genomic distance effects
    num_loci = 651
    #window size in Mb [for now make it as large as possible
    offset = data_combined['mid_position_Mb'][-1] - data_combined['mid_position_Mb'][0]
    #matrices where center [650, 650] is the locus containing gene(s)
    distmap_on = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    distmap_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    rirjmap_on = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    rirjmap_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    counts_on = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    counts_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))

    for i in tqdm(range(len(data_combined['dna_zxys']))):
        distmap = distmap_combined_list[i] #651 by 651 matrix
        rirjmap = rirjmap_combined_list[i] #651 by 651 matrix
        countmap = np.ones_like(distmap)
        #dont count NaNs in average. nans in distmap should be same as in rirjmap
        countmap[np.isnan(distmap)] = 0
        #convert nans to zero
        distmap[np.isnan(distmap)] = 0
        rirjmap[np.isnan(rirjmap)] = 0
        #find all the gene loci that are on vs off
        on_gene_inds = np.where(data_combined['gene_state'][i] == 1)[0]
        off_gene_inds = np.where(data_combined['gene_state'][i] == -1)[0]
        assert(len(on_gene_inds) + len(off_gene_inds) == 80)
        for oni in on_gene_inds:
            #TODO: devise a binning procedure to aggregate distances with fixed binsize
            distmap_on[(num_loci - 1 - oni):(2*num_loci -1 - oni), (num_loci - 1 - oni):(2*num_loci -1 - oni)] += distmap
            rirjmap_on[(num_loci - 1 - oni):(2*num_loci -1 - oni), (num_loci - 1 - oni):(2*num_loci -1 - oni)] += rirjmap
            #keep track of the number of distance computations in each pixel
            counts_on[(num_loci - 1 - oni):(2*num_loci -1 - oni), (num_loci - 1 - oni):(2*num_loci -1 - oni)] += countmap
        for offi in off_gene_inds:
            distmap_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += distmap
            rirjmap_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += rirjmap
            #keep track of the number of distance computations in each pixel
            counts_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += countmap

    #mean
    distmap_off /= counts_off
    rirjmap_off /= counts_off
    distmap_on /= counts_on
    rirjmap_on /= counts_on
    np.save('distmap_off.npy', distmap_off)
    np.save('rirjmap_off.npy', rirjmap_off)
    np.save('distmap_on.npy', distmap_on)
    np.save('rirjmap_on.npy', rirjmap_on)

def compute_OFF_gene_pileups():
    with open('data_combined_chr21.pkl', 'rb') as f:
        data_combined = pickle.load(f)
    with open('rirj_combined_list.pkl', 'rb') as f:
        rirjmap_combined_list = pickle.load(f)
    with open('distmap_combined_list.pkl', 'rb') as f:
        distmap_combined_list = pickle.load(f)
    # normalize genomic distance effects
    num_loci = 651
    distmap_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    rirjmap_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))
    counts_off = np.zeros((2*num_loci - 1, 2*num_loci - 1))

    for i in tqdm(range(len(data_combined['dna_zxys']))):
        distmap = distmap_combined_list[i] #651 by 651 matrix
        rirjmap = rirjmap_combined_list[i] #651 by 651 matrix
        countmap = np.ones_like(distmap)
        #dont count NaNs in average. nans in distmap should be same as in rirjmap
        countmap[np.isnan(distmap)] = 0
        #convert nans to zero
        distmap[np.isnan(distmap)] = 0
        rirjmap[np.isnan(rirjmap)] = 0
        #find all the gene loci that are on vs off
        off_gene_inds = np.where(data_combined['gene_state'][i] == -1)[0]
        for offi in off_gene_inds:
            distmap_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += distmap
            rirjmap_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += rirjmap
            #keep track of the number of distance computations in each pixel
            counts_off[(num_loci - 1 - offi):(2*num_loci -1 - offi), (num_loci - 1 - offi):(2*num_loci -1 - offi)] += countmap

    #mean
    distmap_off /= counts_off
    rirjmap_off /= counts_off
    np.save('distmap_off.npy', distmap_off)
    np.save('rirjmap_off.npy', rirjmap_off)

if __name__ == "__main__":
    compute_OFF_gene_pileups()

