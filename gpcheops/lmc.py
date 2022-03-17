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

def linear_decorrelation(tim, fl, fle, params, plan_param_priors, t14, lin_priors=None, oot_method='single', sampler='dynesty', nthreads=None, save=True, out_path=os.getcwd()):
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
    lin_priors : dict
        This to be used if one wants to provide informative priors to the linear model hyperparameters
        Default is None (i.e., uninformative priors will be used)
        "manual_gp_priors" from gpcheops.utils to be used when generating the dictionary
    oot_method : str
        Which method to be use for selecting out-of-transit/eclipse (or joint) points
        default is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
    sampler : str
        Sampler to use, default is dynesty
        If dynesty or dynamic_dynesty is to be used, one can set nthreads
    nthreads : Nos of threads to used in dynesty, or in dynamic dynesty
        Default is None
    save : bool
        Boolean on whether to save results (figure & decorrelated photometry) or not
        Default is True
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
    # ln_par is a numpy array of size (len(tim), nos. of params)
    # This can directly be provided as a linear regressor to the data

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
    param_oot[instrument] = ln_par[mask]

    ## Priors for out-of-transit analysis
    ### Instrumental priors:
    par_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
    dist_ins = ['fixed', 'normal', 'loguniform']
    hyper_ins = [1.0, [0., 0.1], [0.1, 10000.]]
    ### Priors for linear model parameters:
    if lin_priors == None:
        par_lin, dist_lin, hyper_lin = [], [], []
        for i in range(ln_par.shape[1]):
            par_lin = par_lin + ['theta' + str(i) + '_' + instrument]
            dist_lin = dist_lin + ['uniform']
            hyper_lin = hyper_lin + [-5.0, 5.0]
    else:
        par_lin = lin_priors['params']
        dist_lin = lin_priors['dists']
        hyper_lin = lin_priors['hyper']
    priors_oot = juliet.utils.generate_priors(par_ins+par_lin, dist_ins+dist_lin, hyper_ins+hyper_lin)

    # Now modelling the out-of-transit data
    data_oot = juliet.load(priors=priors_oot, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, linear_regressors_lc=param_oot,\
         out_folder=out_path + '/juliet_'+ instrument +'/juliet_oot')
    if sampler == 'dynamic_dynesty' or sampler == 'dynamic dynesty':
        res_oot = data_oot.fit(sampler = 'dynamic_dynesty', bound = 'single', n_effective = 100, use_stop = False, nthreads = nthreads, verbose=True)
    else:
        res_oot = data_oot.fit(sampler=sampler, nthreads=nthreads, n_live_points=500, verbose=True)

    ### Now the whole dataset
    ## First of all, data
    tim_full, fl_full, fle_full, param_full = {}, {}, {}, {}
    tim_full[instrument], fl_full[instrument], fle_full[instrument] = tim, fl, fle
    param_full[instrument] = ln_par
    ## Instrumental priors:
    # For mflux:
    post_mflx = res_oot.posteriors['posterior_samples']['mflux_' + instrument]
    hyper_ins[1] = [np.median(post_mflx), np.std(post_mflx)]
    # For sigma_w:
    dist_ins[2] = 'truncatednormal'
    post_sig = res_oot.posteriors['posterior_samples']['sigma_w_' + instrument]
    hyper_ins[2] = [np.median(post_sig), np.std(post_sig), hyper_ins[2][0], hyper_ins[2][1]]
    ## Linear model parameters:
    for i in range(len(par_lin)):
        if dist_lin[i] != 'fixed':
            dist_lin[i] = 'truncatednormal'
            post_lin = res_oot.posteriors['posterior_samples'][par_lin[i]]
            hyper_lin[i] = [np.median(post_lin), np.std(post_lin), hyper_lin[i][0], hyper_lin[i][1]]
    ## Planetary priros:
    # Planetary parameters
    params_P, dist_P, hyper_P = list(plan_param_priors.keys()), [], []
    for k in plan_param_priors.keys():
        dist_P.append(plan_param_priors[k]['distribution'])
        hyper_P.append(plan_param_priors[k]['hyperparameters'])
    ## Total priros:
    priors_full = juliet.utils.generate_priors(params_P+par_ins+par_lin, dist_P+dist_ins+dist_lin, hyper_P+hyper_ins+hyper_lin)

    # Now modelling the out-of-transit data
    data_full = juliet.load(priros=priors_full, t_lc=tim_full, y_lc=fl_full, yerr_lc=fle_full, linear_regressors_lc=param_full,\
        out_folder=out_path + '/juliet_'+ instrument +'/juliet_full')
    if sampler == 'dynamic_dynesty' or sampler == 'dynamic dynesty':
        res_full = data_full.fit(sampler = 'dynamic_dynesty', bound = 'single', n_effective = 100, use_stop = False, nthreads = nthreads, verbose=True)
    else:
        res_full = data_full.fit(sampler=sampler, nthreads=nthreads, n_live_points=500, verbose=True)

    # First evaluating the model
    model, model_uerr, model_derr, comps = res_full.lc.evaluate(instrument, return_err=True, return_components=True, all_samples=True)
    oot_flux = np.median(1./(1. + res_full.posteriors['posterior_samples']['mflux_' + instrument]))
    if save:
        # Making a plot of full model
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim_full[instrument], fl_full[instrument], yerr=fle_full[instrument], fmt='.', alpha=0.3)
        ax1.plot(tim_full[instrument], comps['lm']+oot_flux, c='gray', lw=3, zorder=150, alpha=0.7)
        ax1.plot(tim_full[instrument], model, c='k', zorder=100)
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim_full[instrument]), np.max(tim_full[instrument]))
        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim_full[instrument], (fl_full[instrument]-model)*1e6, yerr=fle_full[instrument]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--', zorder=10)
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim_full[instrument]), np.max(tim_full[instrument]))

        plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full/full_model.png')
        plt.close(fig)

        # Only transit model
        # Making a plot
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim_full[instrument], (fl_full[instrument]-comps['lm']) - oot_flux + 1, yerr=fle_full[instrument], fmt='.', alpha=0.3)
        ax1.plot(tim_full[instrument], (model-comps['lm']) - oot_flux + 1, c='k', zorder=100)
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim_full[instrument]), np.max(tim_full[instrument]))
        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim_full[instrument], (fl_full[instrument]-model)*1e6, yerr=fle_full[instrument]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--', zorder=10)
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim_full[instrument]), np.max(tim_full[instrument]))

        if transit:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full/transit_model.png')
        elif eclipse:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full/eclipse_model.png')
        else:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full/transit_eclipse_model.png')
        plt.close(fig)