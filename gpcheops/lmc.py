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

def roll_parameters(roll, degree):
    ab = np.zeros(len(roll))
    for i in range(degree):
        sin1 = np.sin(np.deg2rad((i+1)*roll))
        cos1 = np.cos(np.deg2rad((i+1)*roll))
        ab = np.vstack((ab, sin1))
        ab = np.vstack((ab, cos1))
    return ab[1:]

def linear_decorrelation(tim, fl, fle, params, plan_param_priors, t14, oot_method='single', out_path=os.getcwd()):
    """
    This function will perform transit/eclipse/joint lightcurve analysis
    using linear decorrelation against given set of parameters.
    --------------------------------------------------------------------
    Parameters:
    -----------
    tim, fl, fle : dict, dict, dict
        transit lightcurve data; time, flux and errors in flux
        keys of these dictinaries should be name of the instrument
    params : dict
        Dictinary containing the decorrelation parameters.
        Key should be the name of the parameter
        Note: If you want to provide roll angle as a parameter, you must use
              "ROLL_N", "roll_N", or "Role_N" as the key, where N is the number of degress
    plan_param_priors : dict
        juliet readable planetary priors
    t14 : float
        Transit/eclipse duration
    oot_method : str
        Which method to be use for selecting out-of-transit/eclipse (or joint) points
        default is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
    out_path : str
        Output path for the resultant files
        Default is the present working directory
    """
    ### Essential planetary parameters
    try:
        T0 = plan_param_priors['t0_p1']['hyperparameters'][0]
    except:
        T0 = plan_param_priors['t0_p1']['hyperparameters']
    try:
        per4 = plan_param_priors['P_p1']['hyperparameters'][0]
    except:
        per4 = plan_param_priors['P_p1']['hyperparameters']
    
    ### Indentify the data
    instrument = list(tim.keys())[0]
    tim, fl, fle = tim[instrument], fl[instrument], fle[instrument]

    ### Let's first do the out-of-the-transit analysis.
    # Masking in-transit points
    eclipse = False
    transit = False
    for j in plan_param_priors.keys():
        if j[0:2] == 'fp':
            eclipse = True
        if j[0:2] == 'q1':
            transit = True
    if oot_method == 'single':
        if eclipse and not transit:
            mask = np.where(tim > (T0 + (per4/2) + (t14/2)))[0]
            mask = np.hstack((np.where(tim < (T0 + (per4/2) - (t14/2)))[0], mask))
        elif transit and not eclipse:
            mask = np.where(tim > (T0 + (t14/2)))[0]
            mask = np.hstack((np.where(tim < (T0 - (t14/2)))[0], mask))
        elif transit and eclipse:
            mask = np.where(tim > (T0 + (t14/2)))[0]
            mask = np.hstack((np.where(tim < (T0 - (t14/2)))[0], mask))
            mask = np.hstack((np.where(tim < (T0 + (per4/2) + (t14/2)))[0], mask))
            mask = np.hstack((np.where(tim < (T0 + (per4/2) - (t14/2)))[0], mask))
    elif oot_method == 'multi':
        phs_t = juliet.utils.get_phases(tim, per4, T0)
        phs_e = juliet.utils.get_phases(tim, per4, (T0+(per4/2)))
        if eclipse and not transit:
            mask = np.where(np.abs(phs_e*per4) >= t14)[0]
        elif transit and not eclipse:
            mask = np.where(np.abs(phs_t*per4) >= t14)[0]
        elif transit and eclipse:
            mask = np.where((np.abs(phs_e*per4) >= t14)&(np.abs(phs_t*per4) >= t14))[0]
    else:
        raise Exception('Method to discard out-of-transit/eclipse points can only be "single" or "multi".')
    
    # Out-of-transit data
    tim_oot, fl_oot, fle_oot, param_oot = {}, {}, {}, {}
    tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tim[mask], fl[mask], fle[mask]
    
    # Now, it could be possible that there are more than one parameters
    # are there for linear detrending. All of them are stored in form of
    # a dictionary --- we want to save them in form of a numpy array.
    lst_param = list(params.keys())
    if len(lst_param) == 1:
        if lst_param[0][0:4] == 'ROLL' or lst_param[0][0:4] == 'roll' or lst_param[0][0:4] == 'Roll':
            ln_par = roll_parameters(params[lst_param[0]], int(lst_param[0].split('_')[1]))
            ln_par = np.transpose(ln_par)
        else:
            ln_par = params[lst_param[0]]
            ln_par = ln_par.reshape((len(ln_par), 1))
    else:
        ln_par = np.zeros(len(tim))
        for i in range(len(lst_param)):
            if lst_param[i][0:4] == 'ROLL' or lst_param[i][0:4] == 'roll' or lst_param[i][0:4] == 'Roll':
                ln_par1 = roll_parameters(params[lst_param[i]], int(lst_param[i].split('_')[1]))
                ln_par = np.vstack((ln_par, ln_par1))
            else:
                ln_par = np.vstack((ln_par, params[lst_param[i]]))
        ln_par = np.transpose(ln_par[1:])