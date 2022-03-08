import numpy as np
import matplotlib.pyplot as plt
import juliet
import matplotlib.gridspec as gd
from astropy.table import Table
import os
from path import Path
from glob import glob
import pickle
from scipy.interpolate import CubicSpline
import corner
from matplotlib.gridspec import GridSpec

def corner_plot(folder, planet_only=False):
    """
    This function will generate corner plots of posterios
    in a given folder
    -----------------------------------------------------
    Parameters:
    -----------
    folder : str
        Path of the folder where the .pkl file is located
    planet_only : bool
        Boolean on whether to make corner plot of only
        planetary parameters
        Default is False
    -----------
    return
    -----------
    corner plot : .pdf file
        stored inside folder directory
    """
    pcl = glob(folder + '/*.pkl')[0]
    post = pickle.load(open(pcl, 'rb'), encoding='latin1')
    p1 = post['posterior_samples']
    lst = []
    if not planet_only:
        for i in p1.keys():
            gg = i.split('_')
            if 'p1' in gg or 'mflux' in gg or 'sigma' in gg or 'GP' in gg or 'mdilution' in gg or 'q1' in gg or 'q2' in gg:
                lst.append(i)
    else:
        for i in p1.keys():
            gg = i.split('_')
            if 'p1' in gg or 'q1' in gg or 'q2' in gg:
                lst.append(i)
    if 't0' in lst[0].split('_'):
        t01 = np.floor(p1[lst[0]][0])
        cd = p1[lst[0]] - t01
        lst[0] = lst[0] + ' - ' + str(t01)
    elif 'fp' in lst[0].split('_'):
        cd = p1[lst[0]]*1e6
        lst[0] = lst[0] + ' (in ppm)'
    else:
        cd = p1[lst[0]]
    for i in range(len(lst)-1):
        if 't0' in lst[i+1].split('_'):
            t02 = np.floor(p1[lst[i+1]][0])
            cd1 = p1[lst[i+1]] - t02
            cd = np.vstack((cd, cd1))
            lst[i+1] = lst[i+1] + ' - ' + str(t02)
        elif 'fp' in lst[i+1].split('_'):
            cd = np.vstack((cd, p1[lst[i+1]]*1e6))
            lst[i+1] = lst[i+1] + ' (in ppm)'
        else:
            cd = np.vstack((cd, p1[lst[i+1]]))
    data = np.transpose(cd)
    value = np.median(data, axis=0)
    ndim = len(lst)
    fig = corner.corner(data, labels=lst)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i,i]
        ax.axvline(value[i], color = 'r')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value[xi], color = 'r')
            ax.axhline(value[yi], color = 'r')
            ax.plot(value[xi], value[yi], 'sr')

    fig.savefig(folder + "/corner.pdf")

def correlation_plot(params, flx, flxe, out_folder=os.getcwd()):
    """
    This function will generate trend of flux 
    with various decorrelation parameters
    -----------------------------------------
    Parameters:
    -----------
    params : dict
        dictionary containing decorrelation parameters
    flx : dict
        dictionary containing flux
    flxe : dict
        dictionary containing errors in flux
    out_folder : str
        location where to save the plots
    -----------
    return
    -----------
    .png file
        saved plot
    """
    # Decorrelation parameters
    pnames = list(params.keys())
    nms = len(pnames)
    # Instrument name
    instrument = list(flx.keys())[0]

    fig = plt.figure(figsize=(16,9))
    gs = GridSpec(nms, 1)#, width_ratios=[1, 2], height_ratios=[4, 1])
    for i in range(nms):
        ax1 = fig.add_subplot(gs[i])
        ax1.errorbar(params[pnames[i]], flx[instrument], yerr=flxe[instrument], fmt='.', label=pnames[i])
        ax1.set_ylabel('Trend with ' + pnames[i])

    plt.savefig(out_folder + '/correlation_plot.png')

def manual_gp_priors(params, dists, hyper):
    """
    This function creates GP dictionary to give inputs to
    single_param_decorr() function of gpcheops
    -----------------------------------------------------
    Parameters:
    -----------
    params : list
        List containing GP parameters' names
    dists : list
        Distribution of the GP parameters
    hyper : list
        List containing prior values
    -----------
    return
    -----------
    dict
        Dictionary that can be ingested to single_param_decorr()
    """
    gp_priors = {}
    gp_priors['params'] = params
    gp_priors['dists'] = dists
    gp_priors['hyper'] = hyper
    return gp_priors

def manual_multi_gp_priors(decorr_lst, gp_prs):
    """
    This function returns a dictionary that can be provided to
    multiple_params_decorr() function of gpcheops
    ----------------------------------------------------------
    Parameters:
    -----------
    decorr_lst : list
        A list containing the names of decorrelation parameters
    gp_prs : list
        A list containing dictionaries that have GP priors given to the decorrelation parameters
        Use manual_gp_priors() function of gpcheops.utils to generate these dictionaries
        One can also include strings to use default GP priors from ExM, QP or SHO.
    -----------
    return
    -----------
    dict
        A dictionary that can be ingested to the multiple_params_decorr()
    """
    mult_gp_priors = {}
    for i in range(len(decorr_lst)):
        mult_gp_priors[decorr_lst[i]] = gp_prs[i]
    return mult_gp_priors


def tdur(per, ar, rprs, bb):
    """
    To compute transit/eclipse duration from
    Period, a/R*, Rp/R* and b
    ----------------------------------------
    Parameters:
    -----------
    per : float, or numpy.ndarray
        Orbital period of the planet
    aR : float, or numpy.ndarray
        Scaled semi-major axis, a/R*
    rprs : float, or numpy.ndarray
        Planet-to-star radius ratio, Rp/R*
    bb : float, or numpy.ndarray
        Impact parameter
    -----------
    return
    -----------
    float, or numpy.ndarray
        Transit duration, in days
    """
    ab = per/np.pi
    cd = (1+rprs)**2 - (bb**2)
    ef = 1 - ((bb/ar)**2)
    br1 = (1/ar)*(np.sqrt(cd/ef))
    tt = ab*np.arcsin(br1)
    return tt


def tau(per, ar, rprs, bb):
    """
    To compute ingress/egress duration from
    Period, a/R*, Rp/R* and b
    ----------------------------------------
    Parameters:
    -----------
    per : float, or numpy.ndarray
        Orbital period of the planet
    aR : float, or numpy.ndarray
        Scaled semi-major axis, a/R*
    rprs : float, or numpy.ndarray
        Planet-to-star radius ratio, Rp/R*
    bb : float, or numpy.ndarray
        Impact parameter
    -----------
    return
    -----------
    float, or numpy.ndarray
        Transit duration, in days
    """
    ab = per/np.pi
    bc = 1/np.sqrt(1 - bb**2)
    xy = ab*bc*rprs/ar
    return xy