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

def gp_decorr(tim, fl, fle, param, plan_params, t14, GP='ExM', out_path=os.getcwd(), sampler='dynesty', oot_method='single', nthreads=None, verbose=True):
    """
    This function is to perform GP detrending against one parameter
    --------------------------------------------------------
    Parameters:
    -----------
    tim, fl, fle : dict, dict, dict
        transit lightcurve data; time, flux and error in flux
    param : dict
        decorrelation parameter
    plan_params : dict
        juliet readable priors
        juliet will identify which model (batman or catwoman or eclipse or joint)
        is to be used
    t14 : float
        transit duration
    GP : str or dict
        -- On which GP kernel to use. ExM for Exponential-Matern kernel
           QP for quasi-periodic kernel, SHO for Simple Harmonic Oscillator kernel
        -- If the type of the inpute provided in dict, then it means that the user
           decided to provide GP priors manually instead of the default ones.
           Trick is to use, "manual_gp_priors()" from gpcheops.utils, to provide
           name of the GP parameters, distribution and their values, and use output 
           from this function here.
        Default is ExM

    out_path : str
        output path of the analysed files
        note that everything will be saved in different folders
        which would be sub-folders of a folder called juliet
        default is the present working directory
    sampler : str
        sampler to be used in posterior estimation
        a valid choices are 'multinest', 'dynesty', 'dynamic_dynesty', or 'ultranest'
        default is 'dynesty'
    save : bool
        boolean on whether to save decorrelated photometry
        and plots.
        decorrelated photometry is original photometry minus fitted gp model
        first, second and third column contain time, flux and error in flux
        Plots contain: 1) trend in flux with the decorrelation parameter
        and the best fitted GP model to it. 2) Full model fitted 3) best 
        fitted transit model.
        default is True
    oot_method : str
        Which method to be use for selecting out-of-transit/eclipse (or joint) points
        default is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
    nthreads : int
        Number of threads needed while using dynesty.
        Default is None.
    verbose : bool
        boolean on whether to print progress of analysis
        default is true
    -----------
    return:
    -----------
    .dat file :
        decorrelated photometry stored in out_folder/juliet_instrument/juliet_full_param
    lnZ : float
        log Bayesian evidence for the analysis
    """
    ### Essential planetary parameters
    try:
        T0 = plan_params['t0_p1']['hyperparameters'][0]
    except:
        T0 = plan_params['t0_p1']['hyperparameters']
    try:
        per4 = plan_params['P_p1']['hyperparameters'][0]
    except:
        per4 = plan_params['P_p1']['hyperparameters']
    ### Indentify the data
    instrument = list(tim.keys())[0]
    tim, fl, fle = tim[instrument], fl[instrument], fle[instrument]
    nm_param = list(param.keys())[0]
    param = param[nm_param]
    
    ### Let's first do the out-of-the-transit analysis.
    # Masking in-transit points
    eclipse = False
    transit = False
    for j in plan_params.keys():
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
    param_oot[instrument] = param[mask]
    
    # Sorting the data according to the decorrelating parameter
    tt = Table()
    tt['tim'], tt['fl'], tt['fle'] = tim_oot[instrument], fl_oot[instrument], fle_oot[instrument]
    tt['param'] = param_oot[instrument]
    tt.sort('param')
    tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tt['tim'], tt['fl'], tt['fle']
    param_oot[instrument] = tt['param']
    
    ## Defining priors
    # Instrumental parameters
    params_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
    dist_ins = ['fixed', 'normal', 'loguniform']
    hyper_ins = [1., [0., 0.1], [0.1, 10000.]]
    # GP parameters
    if GP == 'ExM':
        params_gp = ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform']
        hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
    elif GP == 'QP':
        params_gp = ['GP_B_' + instrument, 'GP_C_' + instrument, 'GP_L_' + instrument, 'GP_Prot_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform','loguniform']
        hyper_gp = [[1e-5,1e3], [1e-5,1e4], [1e-3, 1e3], [1.,1e2]]
    elif GP == 'SHO':
        params_gp = ['GP_S0_' + instrument, 'GP_omega0_' + instrument, 'GP_Q_' + instrument]
        dist_gp = ['uniform', 'uniform', 'fixed']
        hyper_gp = [[np.exp(-40.), np.exp(0.)], [np.exp(-10.), np.exp(10.)], np.exp(1/np.sqrt(2))]
    elif type(GP) == dict:
        params_gp = GP['params']
        dist_gp = GP['dists']
        hyper_gp = GP['hyper']
    else:
        raise Exception('GP method can only be ExM, QP or SHO.\n Or it can be a dictionary')
    # Total priors
    params_gp_only = params_ins + params_gp
    dist_gp_only = dist_ins + dist_gp
    hyper_gp_only = hyper_ins + hyper_gp
    # Populating prior dict
    priors = juliet.utils.generate_priors(params_gp_only, dist_gp_only, hyper_gp_only)

    ## Running GP only fit
    data = juliet.load(priors=priors, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, GP_regressors_lc=param_oot,\
         out_folder=out_path + '/juliet_'+ instrument +'/juliet_' + nm_param)
    if sampler == 'dynamic_dyesty' or sampler == 'dynamic dynesty':
        res_gp_only = data.fit(sampler = 'dynamic_dynesty', bound = 'single', n_effective = 100, use_stop = False, nthreads = nthreads, verbose=True)
    else:
        res_gp_only = data.fit(sampler = sampler, n_live_points=500, nthreads=nthreads, verbose = verbose)

    results_full = res_gp_only
    tim, fl, fle, param = tim_oot, fl_oot, fle_oot, param_oot

    ### Evaluating the fitted model
    # juliet best fit model
    model, model_uerr, model_derr, comps = results_full.lc.evaluate(instrument, return_err=True, return_components=True, all_samples=True)
    # juliet best fit gp model
    gp_model = results_full.lc.model[instrument]['GP']
    gp_model_uerr = results_full.lc.model[instrument]['GP_uerror']
    gp_model_derr = results_full.lc.model[instrument]['GP_lerror']
    # juliet best fit transit model and its errors
    transit_model = results_full.lc.model[instrument]['deterministic']
    transit_model_err = results_full.lc.model[instrument]['deterministic_errors']

    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(param[instrument], (fl[instrument]-transit_model), yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.plot(param[instrument], gp_model, c='k', zorder=100)
    ax1.fill_between(param[instrument], model_derr-transit_model, model_uerr-transit_model, color='k', alpha=0.3, zorder=100)
    ax1.set_ylabel('Trend with ' + nm_param)
    ax1.set_xlim(np.min(param[instrument]), np.max(param[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(param[instrument], (fl[instrument]-gp_model-transit_model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--')
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel(nm_param)
    ax2.set_xlim(np.min(param[instrument]), np.max(param[instrument]))

    # Saving the figure
    plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_' + nm_param + '/decorr_' + nm_param +'.png')
    plt.close(fig)
    
    ## Sorting again in time ascending order to make lightcurve plots
    tt3 = Table()
    tt3['tim'] = tim[instrument]
    tt3['fl'] = fl[instrument]
    tt3['fle'] = fle[instrument]
    tt3['model'] = model
    tt3['model_ue'] = model_uerr
    tt3['model_de'] = model_derr
    tt3['gp_mod'] = gp_model
    tt3['gp_mod_ue'] = gp_model_uerr
    tt3['gp_mod_de'] = gp_model_derr
    tt3['tran_mod'] = transit_model
    tt3['tran_mod_err'] = transit_model_err

    tt3.sort('tim')

    tim[instrument] = tt3['tim']
    fl[instrument] = tt3['fl']
    fle[instrument] = tt3['fle']
    model = tt3['model']
    model_uerr = tt3['model_ue']
    model_derr = tt3['model_de']
    gp_model = tt3['gp_mod']
    gp_model_uerr = tt3['gp_mod_ue']
    gp_model_derr = tt3['gp_mod_de']
    transit_model = tt3['tran_mod']
    transit_model_err = tt3['tran_mod_err']

    ## Making lightcurves
    # Full model
    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.plot(tim[instrument], model, c='k', zorder=100)
    ax1.fill_between(tim[instrument], model_derr, model_uerr, color='k', alpha=0.3, zorder=100)
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--')
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))

    plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_' + nm_param + '/full_model_' + nm_param + '.png')
    plt.close(fig)
    
    fac = 1/np.max(transit_model)

    ## Decorrelating!!
    tim1, fl1, fle1, resid1 = tim[instrument], (fl[instrument]-gp_model)*fac, fle[instrument], (fl[instrument]-model)*1e6
    err7 = (model_uerr-model_derr)/2
    fle2 = (np.sqrt((fle1**2) + (err7**2)))*fac
    f1 = open(out_path + '/juliet_'+ instrument +'/juliet_' + nm_param + '/' + nm_param + '_decorrelated_photometry.dat','w')
    for i in range(len(tim[instrument])):
        f1.write(str(tim1[i]) + '\t' + str(fl1[i]) + '\t' + str(fle2[i]) + '\t' + str(resid1[i]) + '\n')
    f1.close()

    return results_full.posteriors['lnZ']

def roll_parameters(roll, degree):
    ab = np.zeros(len(roll))
    for i in range(degree):
        sin1 = np.sin(np.deg2rad((i+1)*roll))
        cos1 = np.cos(np.deg2rad((i+1)*roll))
        ab = np.vstack((ab, sin1))
        ab = np.vstack((ab, cos1))
    return ab[1:]

def linear_decorr(tim, fl, fle, params, plan_param_priors, t14, lin_priors=None, oot_method='single', sampler='dynesty', nthreads=None, out_path=os.getcwd()):
    """
    This function will perform linear detrending against the given parameters.
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
            hyper_lin = hyper_lin + [[-5.0, 5.0]]
    else:
        par_lin = lin_priors['params']
        dist_lin = lin_priors['dists']
        hyper_lin = lin_priors['hyper']
    priors_oot = juliet.utils.generate_priors(par_ins+par_lin, dist_ins+dist_lin, hyper_ins+hyper_lin)

    # Now modelling the out-of-transit data
    data_oot = juliet.load(priors=priors_oot, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, linear_regressors_lc=param_oot,\
         out_folder=out_path + '/juliet_'+ instrument +'/juliet_lin')
    if sampler == 'dynamic_dynesty' or sampler == 'dynamic dynesty':
        res_oot = data_oot.fit(sampler = 'dynamic_dynesty', bound = 'single', n_effective = 100, use_stop = False, nthreads = nthreads, verbose=True)
    else:
        res_oot = data_oot.fit(sampler=sampler, nthreads=nthreads, n_live_points=500, verbose=True)
    res_full = res_oot
    tim_full, fl_full, fle_full = tim_oot, fl_oot, fle_oot

    # First evaluating the model
    model, model_uerr, model_derr, comps = res_full.lc.evaluate(instrument, return_err=True, return_components=True, all_samples=True)
    oot_flux = np.median(1./(1. + res_full.posteriors['posterior_samples']['mflux_' + instrument]))
        
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

    plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_lin/full_model.png')
    plt.close(fig)

    # Saving the decorrelated photometry:
    tim1, fl1, fle1, resid1 = tim_full[instrument], fl_full[instrument]-comps['lm']-oot_flux+1, fle_full[instrument], (fl_full[instrument]-model)*1e6
    err1 = (model_uerr-model_uerr)/2
    fle2 = np.sqrt((fle1**2) + (err1**2))
    f1 = open(out_path + '/juliet_'+ instrument +'/juliet_lin/LINEAR_decorrelated_photometry.dat', 'w')
    for i in range(len(tim1)):
        f1.write(str(tim1[i]) + '\t' + str(fl1[i]) + '\t' + str(fle2[i]) + '\t' + str(resid1[i]) + '\n')
    f1.close()

def linear_gp(tim, fl, fle, lin_params, GP_param, plan_params, t14, lin_priors=None, GP='ExM', out_path=os.getcwd(), sampler='dynesty', oot_method='single', nthreads=None, verbose=True):
    """
    This function do the transit light curve analysis with
    lightcurve decorrelation done against a given parameter.
    For a single visit
    --------------------------------------------------------
    Parameters:
    -----------
    tim, fl, fle : dict, dict, dict
        transit lightcurve data; time, flux and error in flux
    lin_params: dict
        Linear decorrelation parameters
    GP_param : dict
        decorrelation parameter against which GP decorrelation has to be performed
    plan_params : dict
        juliet readable priors
        juliet will identify which model (batman or catwoman or eclipse or joint)
        is to be used
    t14 : float
        transit duration
    lin_priors : dict
        This to be used if one wants to provide informative priors to the linear model hyperparameters
        Default is None (i.e., uninformative priors will be used)
        "manual_gp_priors" from gpcheops.utils to be used when generating the dictionary
    GP : str or dict
        -- On which GP kernel to use. ExM for Exponential-Matern kernel
           QP for quasi-periodic kernel, SHO for Simple Harmonic Oscillator kernel
        -- If the type of the inpute provided in dict, then it means that the user
           decided to provide GP priors manually instead of the default ones.
           Trick is to use, "manual_gp_priors()" from gpcheops.utils, to provide
           name of the GP parameters, distribution and their values, and use output 
           from this function here.
        Default is ExM
    out_path : str
        output path of the analysed files
        note that everything will be saved in different folders
        which would be sub-folders of a folder called juliet
        default is the present working directory
    sampler : str
        sampler to be used in posterior estimation
        a valid choices are 'multinest', 'dynesty', 'dynamic_dynesty', or 'ultranest'
        default is 'dynesty'
    save : bool
        boolean on whether to save decorrelated photometry
        and plots.
        decorrelated photometry is original photometry minus fitted gp model
        first, second and third column contain time, flux and error in flux
        Plots contain: 1) trend in flux with the decorrelation parameter
        and the best fitted GP model to it. 2) Full model fitted 3) best 
        fitted transit model.
        default is True
    oot_method : str
        Which method to be use for selecting out-of-transit/eclipse (or joint) points
        default is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
    nthreads : int
        Number of threads to be used when using dynesty
        Default is None.
    verbose : bool
        boolean on whether to print progress of analysis
        default is true
    -----------
    return:
    -----------
    .dat file :
        decorrelated photometry stored in out_folder/juliet_instrument/juliet_full_param
    lnZ : float
        log Bayesian evidence for the analysis
    """
    ### Essential planetary parameters
    try:
        T0 = plan_params['t0_p1']['hyperparameters'][0]
    except:
        T0 = plan_params['t0_p1']['hyperparameters']
    try:
        per4 = plan_params['P_p1']['hyperparameters'][0]
    except:
        per4 = plan_params['P_p1']['hyperparameters']
    ### Indentify the data
    instrument = list(tim.keys())[0]
    tim, fl, fle = tim[instrument], fl[instrument], fle[instrument]
    nm_param = list(GP_param.keys())[0]
    GP_param = GP_param[nm_param]

    # Now, it could be possible that there are more than one parameters
    # are there for linear detrending. All of them are stored in form of
    # a dictionary --- we want to save them in form of a numpy array.
    lst_param = list(lin_params.keys())
    if len(lst_param) == 1:
        if lst_param[0][0:4] == 'ROLL':
            ln_par = roll_parameters(lin_params[lst_param[0]], int(lst_param[0].split('_')[1]))
            ln_par = np.transpose(ln_par)
        else:
            ln_par = lin_params[lst_param[0]]
            ln_par = ln_par.reshape((len(ln_par), 1))
    else:
        ln_par = np.zeros(len(tim))
        for i in range(len(lst_param)):
            if lst_param[i][0:4] == 'ROLL' or lst_param[i][0:4] == 'roll' or lst_param[i][0:4] == 'Roll':
                ln_par1 = roll_parameters(lin_params[lst_param[i]], int(lst_param[i].split('_')[1]))
                ln_par = np.vstack((ln_par, ln_par1))
            else:
                ln_par = np.vstack((ln_par, lin_params[lst_param[i]]))
        ln_par = np.transpose(ln_par[1:])
    # ln_par is a numpy array of size (len(tim), nos. of params)
    # This can directly be provided as a linear regressor to the data
    
    ### Let's first do the out-of-the-transit analysis.
    # Masking in-transit points
    eclipse = False
    transit = False
    for j in plan_params.keys():
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
    tim_oot, fl_oot, fle_oot, GP_param_oot, lin_param_oot = {}, {}, {}, {}, {}
    tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tim[mask], fl[mask], fle[mask]
    GP_param_oot[instrument] = GP_param[mask]
    lin_param_oot[instrument] = ln_par[mask]
    
    # Sorting the data according to the decorrelating parameter
    tt = Table()
    tt['tim'], tt['fl'], tt['fle'] = tim_oot[instrument], fl_oot[instrument], fle_oot[instrument]
    tt['gp_param'] = GP_param_oot[instrument]
    tt['lin_param'] = lin_param_oot[instrument]
    tt.sort('gp_param')
    tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tt['tim'], tt['fl'], tt['fle']
    GP_param_oot[instrument] = tt['gp_param']
    lin_param_oot[instrument] = tt['lin_param']
    
    ## Defining priors
    # Instrumental parameters
    params_ins = ['mdilution_' + instrument, 'mflux_' + instrument, 'sigma_w_' + instrument]
    dist_ins = ['fixed', 'normal', 'loguniform']
    hyper_ins = [1., [0., 0.1], [0.1, 10000.]]
    # GP parameters
    if GP == 'ExM':
        params_gp = ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform']
        hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
    elif GP == 'QP':
        params_gp = ['GP_B_' + instrument, 'GP_C_' + instrument, 'GP_L_' + instrument, 'GP_Prot_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform','loguniform']
        hyper_gp = [[1e-5,1e3], [1e-5,1e4], [1e-3, 1e3], [1.,1e2]]
    elif GP == 'SHO':
        params_gp = ['GP_S0_' + instrument, 'GP_omega0_' + instrument, 'GP_Q_' + instrument]
        dist_gp = ['uniform', 'uniform', 'fixed']
        hyper_gp = [[np.exp(-40.), np.exp(0.)], [np.exp(-10.), np.exp(10.)], np.exp(1/np.sqrt(2))]
    elif type(GP) == dict:
        params_gp = GP['params']
        dist_gp = GP['dists']
        hyper_gp = GP['hyper']
    else:
        raise Exception('GP method can only be ExM, QP or SHO.\n Or it can be a dictionary')
    # Linear decorrelation parameters:
    if lin_priors == None:
        par_lin, dist_lin, hyper_lin = [], [], []
        for i in range(ln_par.shape[1]):
            par_lin = par_lin + ['theta' + str(i) + '_' + instrument]
            dist_lin = dist_lin + ['uniform']
            hyper_lin = hyper_lin + [[-5.0, 5.0]]
    else:
        par_lin = lin_priors['params']
        dist_lin = lin_priors['dists']
        hyper_lin = lin_priors['hyper']
    # Total priors
    params_gp_only = params_ins + params_gp + par_lin
    dist_gp_only = dist_ins + dist_gp + dist_lin
    hyper_gp_only = hyper_ins + hyper_gp + hyper_lin
    # Populating prior dict
    priors = juliet.utils.generate_priors(params_gp_only, dist_gp_only, hyper_gp_only)

    ## Running GP only fit
    data = juliet.load(priors=priors, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, GP_regressors_lc=GP_param_oot,\
         linear_regressors_lc=lin_param_oot, out_folder=out_path + '/juliet_'+ instrument +'/juliet_lin-' + nm_param)
    if sampler == 'dynamic_dynesty' or sampler == 'dynamic dynesty':
        res_gp_only = data.fit(sampler = 'dynamic_dynesty', bound = 'single', n_effective = 100, use_stop = False, nthreads = nthreads, verbose=True)
    else:
        res_gp_only = data.fit(sampler = sampler, n_live_points=500, nthreads=nthreads, verbose = verbose)

    results_full = res_gp_only
    tim, fl, fle, GP_param = tim_oot, fl_oot, fle_oot, GP_param_oot

    ### Evaluating the fitted model
    # juliet best fit model
    model, model_uerr, model_derr, comps = results_full.lc.evaluate(instrument, return_err=True, return_components=True, all_samples=True)
    # juliet best fit gp model
    gp_model = results_full.lc.model[instrument]['GP']
    # juliet best fit transit model
    transit_model = results_full.lc.model[instrument]['deterministic']

    # Saving the decorrelation plot
    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(GP_param[instrument], (fl[instrument]-transit_model), yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.plot(GP_param[instrument], gp_model, c='k', zorder=100)
    ax1.fill_between(GP_param[instrument], model_derr-transit_model, model_uerr-transit_model, color='k', alpha=0.3, zorder=100)
    ax1.set_ylabel('Trend with ' + nm_param)
    ax1.set_xlim(np.min(GP_param[instrument]), np.max(GP_param[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(GP_param[instrument], (fl[instrument]-gp_model-transit_model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--')
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel(nm_param)
    ax2.set_xlim(np.min(GP_param[instrument]), np.max(GP_param[instrument]))

    # Saving the figure
    plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_lin-' + nm_param + '/decorr_' + nm_param +'.png')
    plt.close(fig)
    
    ## Sorting again in time ascending order to make lightcurve plots
    tt3 = Table()
    tt3['tim'] = tim[instrument]
    tt3['fl'] = fl[instrument]
    tt3['fle'] = fle[instrument]
    tt3['model'] = model
    tt3['model_uerr'] = model_uerr
    tt3['model_derr'] = model_derr
    tt3['gp_mod'] = gp_model
    tt3['tran_mod'] = transit_model
    tt3['comps_lm'] = comps['lm']

    tt3.sort('tim')

    tim[instrument] = tt3['tim']
    fl[instrument] = tt3['fl']
    fle[instrument] = tt3['fle']
    model = tt3['model']
    model_uerr = tt3['model_uerr']
    model_derr = tt3['model_derr']
    gp_model = tt3['gp_mod']
    transit_model = tt3['tran_mod']
    comps['lm'] = tt3['comps_lm']

    ## Making lightcurves
    # Full model
    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.plot(tim[instrument], model, c='k', zorder=100)
    ax1.fill_between(tim[instrument], model_uerr, model_derr, color='k', alpha=0.3, zorder=100)
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--')
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))

    plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_lin-' + nm_param + '/full_model_' + nm_param + '.png')
    plt.close(fig)

    fac = 1/np.max(model-gp_model-comps['lm'])#1/(1+np.median(mflux))
    mods_err = (model_uerr-model_derr)/2
    flxe = fle[instrument]
    flx_errs = np.sqrt((mods_err**2) + (flxe**2))

    ## Decorrelating!!
    tim1, fl1, fle1, resid1 = tim[instrument], (fl[instrument]-gp_model-comps['lm'])*fac, flx_errs, (fl[instrument]-model)*1e6
    f1 = open(out_path + '/juliet_'+ instrument +'/juliet_lin-' + nm_param + '/' + nm_param + '-lin_decorrelated_photometry.dat','w')
    for i in range(len(tim[instrument])):
        f1.write(str(tim1[i]) + '\t' + str(fl1[i]) + '\t' + str(fle1[i]) + '\t' + str(resid1[i]) + '\n')
    f1.close()

    return results_full.posteriors['lnZ']