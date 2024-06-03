""" Useful plotting functions for post processing analysis. """

import numpy as np
import scipy
import matplotlib
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

slide_width = 11.5
half_slide_width = 5.67
aspect_ratio = 5/7
pres_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 0.5,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'axes.titlesize': 20,
                  'axes.labelsize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'text.usetex': False,
                  'figure.figsize': [half_slide_width, half_slide_width * aspect_ratio],
                  'font.family': 'sans-serif',
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.75,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.75,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}

plt.rcParams.update(pres_params)
cmap_distance = 'magma'
cmap_temps = 'coolwarm'
cmap_contacts = "YlOrRd"
bp_formatter = EngFormatter('b')
norm = LogNorm(vmax=0.1)

def format_ticks(ax, x=True, y=True, rotate=True):
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)

def draw_power_law_triangle(alpha, x0, width, orientation, base=10,
                            x0_logscale=True, label=None, hypotenuse_only=False,
                            label_padding=0.1, text_args={}, ax=None,
                            **kwargs):
    """Draw a triangle showing the best-fit power-law on a log-log scale.

    Parameters
    ----------
    alpha : float
        the power-law slope being demonstrated
    x0 : (2,) array_like
        the "left tip" of the power law triangle, where the hypotenuse starts
        (in log units)
    width : float
        horizontal size in number of major log ticks (default base-10)
    orientation : string
        'up' or 'down', control which way the triangle's right angle "points"
    base : float
        scale "width" for non-base 10
    ax : mpl.axes.Axes, optional

    Returns
    -------
    corner : (2,) np.array
        coordinates of the right-angled corhow to get text outline of the
        triangle
    """
    if x0_logscale:
        x0, y0 = [base**x for x in x0]
    else:
        x0, y0 = x0
    if ax is None:
        ax = plt.gca()
    x1 = x0*base**width
    y1 = y0*(x1/x0)**alpha
    ax.plot([x0, x1], [y0, y1], 'k')
    corner = [x0, y0]
    if not hypotenuse_only:
        if (alpha >= 0 and orientation == 'up') \
                or (alpha < 0 and orientation == 'down'):
            ax.plot([x0, x1], [y1, y1], 'k')
            ax.plot([x0, x0], [y0, y1], 'k')
            # plt.plot lines have nice rounded caps
            # plt.hlines(y1, x0, x1, **kwargs)
            # plt.vlines(x0, y0, y1, **kwargs)
            corner = [x0, y1]
        elif (alpha >= 0 and orientation == 'down') \
                or (alpha < 0 and orientation == 'up'):
            ax.plot([x0, x1], [y0, y0], 'k')
            ax.plot([x1, x1], [y0, y1], 'k')
            # plt.hlines(y0, x0, x1, **kwargs)
            # plt.vlines(x1, y0, y1, **kwargs)
            corner = [x1, y0]
        else:
            raise ValueError(r"Need $\alpha\in\mathbb{R} and orientation\in{'up', "
                             r"'down'}")
    if label is not None:
        xlabel = x0*base**(width/2)
        if orientation == 'up':
            ylabel = y1*base**label_padding
        else:
            ylabel = y0*base**(-label_padding)
        ax.text(xlabel, ylabel, label, horizontalalignment='center',
                verticalalignment='center', **text_args)
    return corner

def plot_imaging_HiC_PC1(data_combined, imaging_pc1, hic_pc1, chrom="2"):
    ## pc1 barplot
    fig, ax = plt.subplots()
    grid = plt.GridSpec(2, 1, height_ratios=[1,1], hspace=0., wspace=0.)
    contact_ax = plt.subplot(grid[0])
    loci_range = data_combined['mid_position_Mb'] * 10**6
    contact_ax.plot(loci_range, imaging_pc1, 'k', lw=1.0)
    contact_ax.fill_between(loci_range, 0, imaging_pc1, where=imaging_pc1 >= 0,
                           color='r')
    contact_ax.fill_between(loci_range, 0, imaging_pc1, where=imaging_pc1 < 0,
                           color='b')
    contact_ax.tick_params('both', 
                pad=1,labelbottom=False)
    contact_ax.set_ylabel('Imaging PC1')
    #contact_ax.set_xticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    # hic-ax
    hic_ax = plt.subplot(grid[1], sharex=contact_ax)
    hic_ax.plot(loci_range, hic_pc1, 'k', lw=1.0)
    hic_ax.fill_between(loci_range, 0, hic_pc1, where=hic_pc1 >= 0, color='r')
    hic_ax.fill_between(loci_range, 0, hic_pc1, where=hic_pc1 < 0, color='b')
    hic_ax.set_ylabel('Hi-C PC1')
    #hic_ax.set_xticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    format_ticks(hic_ax, y=False)
    fig.tight_layout()
    plt.savefig(os.path.join(figure_folder, f'chr{chrom}_pc1_combined.pdf'), transparent=True)
    

#plot mean distance map for chromosome 21 using combined data set
def plot_msd_map(dist, start_position_Mb, end_position_Mb,
                 filename=None, relative=False, squared=False, chrom='2'):
    """ Plot seaborn heatmap where entry (i, j) is the mean distance between beads i and j.
    This version does not include a color bar showing the activity of the monomers.

    Parameters
    ----------
    dist : array-like (N, N)
        matrix containing pairwise mean squared distances between N monomers
    simdir : str or Path
        path to simulation directory containing raw data from which `dist` was computed
    relative : bool
        whether `dist` contains the difference between msds of `simdir` and a reference (eq)
    squared : bool
        whether `dist` contains mean squared distances as opposed to mean distances
    """
    if filename is None:
        filename = f'chr_{chrom}'
    fig, ax = plt.subplots()
    im = plt.imshow(dist, cmap=cmap_distance,
                   extent=[start_position_Mb*10**6, end_position_Mb*10**6,
                          start_position_Mb*10**6, end_position_Mb*10**6])
    plt.colorbar(im)
    #ax.set_xticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    #ax.set_yticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    format_ticks(ax)
    ax.set_title(f'Chr {chrom} Mean Distance', fontsize=18)
    fig.tight_layout()
    if relative:
        plt.savefig(f'../../plots/mean_distance_map_{filename}_relative.pdf')
    else:
        plt.savefig(f'../../plots/mean_distance_map_{filename}.pdf')
    plt.show()

#plot mean distance map for chromosome 21 using combined data set
def plot_contact_map(dist, start_position_Mb, end_position_Mb,
                     filename=None, relative=False, squared=False, chrom='2'):
    """ Plot seaborn heatmap where entry (i, j) is the mean distance between beads i and j.
    This version does not include a color bar showing the activity of the monomers.

    Parameters
    ----------
    dist : array-like (N, N)
        matrix containing pairwise mean squared distances between N monomers
    simdir : str or Path
        path to simulation directory containing raw data from which `dist` was computed
    relative : bool
        whether `dist` contains the difference between msds of `simdir` and a reference (eq)
    squared : bool
        whether `dist` contains mean squared distances as opposed to mean distances
    """
    if filename is None:
        filename = f'chr_{chrom}'
    fig, ax = plt.subplots()
    lognorm = LogNorm(vmin=dist.min(), vmax=dist.max())
    im = plt.imshow(dist, cmap=cmap_contacts, norm=lognorm,
                   extent=[start_position_Mb*10**6, end_position_Mb*10**6,
                          start_position_Mb*10**6, end_position_Mb*10**6])
    plt.colorbar(im)
    #ax.set_xticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    #ax.set_yticks([15*10**6, 25*10**6, 35*10**6, 45*10**6])
    format_ticks(ax)
    ax.set_title(f'Chr {chrom} Contact Map', fontsize=18)
    fig.tight_layout()
    if relative:
        plt.savefig(f'../../plots/contact_map_{filename}_relative.pdf')
    else:
        plt.savefig(f'../../plots/contact_map_{filename}.pdf')
    plt.show()

def expected(contacts, mid_position_Mb, filename, triangle_x0=None, mode='contacts'):
    """ Computing contact probability as a function of distance from diagonal.

    Parameters
    ----------
    counts : array-like (N, N)
        raw integer counts of contacts between pairs of monomers
    nreplicates : int
        total number of simulation snapshots used to compute ensemble average

    Returns
    -------
    Ploop : array-like (N,)
        contact probabilities (averaged over diagonal of contact map)
    sdistances : array-like (N,)
        distances along chain in Kuhn lengths
    nreplicates : int

    """
    N, _ = contacts.shape
    expected = np.zeros(N)
    for i in range(N):
        expected[i] = np.mean(np.diagonal(contacts, offset=i))
    fig, ax = plt.subplots()
    sdistances = np.cumsum(np.diff(mid_position_Mb))
    ax.plot(sdistances, expected[1:])
    if triangle_x0:
        result = scipy.stats.linregress(np.log10(sdistances[0:int(0.6 * len(sdistances))]),
                                        np.log10(expected[1:(int(0.6 * len(sdistances))+1)]))
        width = 1.0
        base = 10
        corner = draw_power_law_triangle(result.slope, triangle_x0, width, 'up', hypotenuse_only=True)
        #text coordinates
        x0, y0 = [base**x for x in triangle_x0]
        x1 = x0*base**(width/4)
        y1 = y0*(x1/x0)**result.slope
        #ydiff = result.slope * (width/2) + result.intercept
        ylo, yhi = ax.get_ylim()
        #center text in the middle of the line
        #text_x = 10**(x + width/2)
        print(yhi - ylo)
        #offset text vertically from line 
        text_y = y1 + 0.1 * (yhi - ylo)
        ax.text(x1, text_y, f'$s^{{{result.slope:.2f}}}$')
    
    format_ticks(ax, y=False, rotate=False)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('genomic distance (Mb)')
    ax.set_ylabel(mode.replace("_", " "))
    ax.set_title(filename.replace("_", " "))
    fig.tight_layout()
    plt.savefig(f'../../plots/{mode}_scaling_{filename}.pdf')