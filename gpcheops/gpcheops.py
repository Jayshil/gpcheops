import numpy as np
import matplotlib.pyplot as plt
import juliet
import matplotlib.gridspec as gd
from astropy.table import Table
import os
from path import Path


def regress(x):
    """
    To normalise x
    (For usage in normalise decorrelation parameter)
    """
    return (x - np.mean(x))/np.sqrt(np.var(x))


def single_param_decorr(tim, fl, fle, param, plan_params, t14, out_path=os.getcwd(), save=True, verbose=True):
    """
    This function do the transit light curve analysis with
    lightcurve decorrelation done against a given parameter.
    For a single visit
    --------------------------------------------------------
    Parameters:
    -----------
    tim, fl, fle : dict, dict, dict
        transit lightcurve data; time, flux and error in flux
    param : dict
        decorrelation parameter
    plan_params : dict
        juliet readable priors
        juliet will identify which model (batman or catwoman)
        is to be used
    t14 : float
        transit duration
    out_path : str
        output path of the analysed files
        note that everything will be saved in different folders
        which would be sub-folders of a folder called juliet
        default is the present working directory
    save : bool
        boolean on whether to save decorrelated photometry
        and plots.
        decorrelated photometry is original photometry minus fitted gp model
        first, second and third column contain time, flux and error in flux
        Plots contain: 1) trend in flux with the decorrelation parameter
        and the best fitted GP model to it. 2) Full model fitted 3) best 
        fitted transit model.
        default is True
    verbose : bool
        boolean on whether to print progress of analysis
        default is true
    -----------
    return:
    -----------
    .dat file :
        decorrelated photometry stored in out_folder/juliet/juliet_full_param
    lnZ : float
        log Bayesian evidence for the analysis
    """
    ### Essential planetary parameters
    T0 = plan_params['t0_p1']['hyperparameters'][0]
    ### Indentify the data
    instrument = list(tim.keys())[0]
    tim, fl, fle = tim[instrument], fl[instrument], fle[instrument]
    nm_param = list(param.keys())[0]
    param = param[nm_param]
    
    
    ### Let's first do the out-of-the-transit analysis.
    # Masking in-transit points
    mask = np.where(tim > (T0 + (t14/2)))[0]
    mask = np.hstack((np.where(tim < (T0 - (t14/2)))[0], mask))
    
    # Out-of-transit data
    tim_oot, fl_oot, fle_oot, param_oot = {}, {}, {}, {}
    tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tim[mask], fl[mask], fle[mask]
    param_oot[instrument] = regress(param[mask])
    
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
    params_gp = ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
    dist_gp = ['loguniform', 'loguniform', 'loguniform']
    hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
    # Total priors
    params_gp_only = params_ins + params_gp
    dist_gp_only = dist_ins + dist_gp
    hyper_gp_only = hyper_ins + hyper_gp
    # Populating prior dict
    priors = juliet.utils.generate_priors(params_gp_only, dist_gp_only, hyper_gp_only)

    ## Running GP only fit
    data = juliet.load(priors=priors, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, GP_regressors_lc=param_oot,\
         out_folder=out_path + '/juliet/juliet_oot_' + nm_param)
    res_gp_only = data.fit(sampler = 'dynesty', n_live_points=500, verbose = verbose)

    ### Now it's time for a full fitting
    # All data is already provided
    # Re-making data dictionary
    tim7, fl7, fle7, param7 = {}, {}, {}, {}
    tim7[instrument], fl7[instrument], fle7[instrument] = tim, fl, fle
    param7[instrument] = param
    tim, fl, fle, param = tim7, fl7, fle7, param7

    # Sorting the data according to the decorrelating parameter
    tt1 = Table()
    tt1['tim'], tt1['fl'], tt1['fle'] = tim[instrument], fl[instrument], fle[instrument]
    tt1['param'] = param[instrument]
    tt1.sort('param')
    tim[instrument], fl[instrument], fle[instrument] = tt1['tim'], tt1['fl'], tt1['fle']
    param[instrument] = tt1['param']

    ## Defining priors
    # We would take instrumental priors from our previous fit
    for i in range(len(params_gp)):
        post1 = res_gp_only.posteriors['posterior_samples'][params_gp[i]]
        mu, sig = np.median(post1), np.std(post1)
        dist_gp[i] = 'truncatednormal'
        hyper_gp[i] = [mu, sig, hyper_gp[i][0], hyper_gp[i][1]]
    # Same goes for mflux and sigma_w
    # For sigma_w_CHEOPS
    dist_ins[2] = 'normal'
    post2 = res_gp_only.posteriors['posterior_samples']['sigma_w_CHEOPS']
    mu, sig = np.median(post2), np.std(post2)
    hyper_ins[2] = [mu, sig]#, hyper_ins[2][0], hyper_ins[2][1]]
    # For mflux
    dist_ins[1] = 'normal'
    post2 = res_gp_only.posteriors['posterior_samples']['mflux_CHEOPS']
    mu, sig = np.median(post2), np.std(post2)
    hyper_ins[1] = [mu, sig]
    # Planetary parameters
    params_P, dist_P, hyper_P = list(plan_params.keys()), [], []
    for k in plan_params.keys():
        dist_P.append(plan_params[k]['distribution'])
        hyper_P.append(plan_params[k]['hyperparameters'])
    # Total priors
    params = params_P + params_ins + params_gp
    dist = dist_P + dist_ins + dist_gp
    hyper = hyper_P + hyper_ins + hyper_gp
    # Prior dictionary
    priors = juliet.utils.generate_priors(params, dist, hyper)

    # Running the whole fit
    data_full = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, GP_regressors_lc=param,\
         out_folder=out_path + '/juliet/juliet_full_' + nm_param)
    results_full = data_full.fit(sampler = 'dynesty', n_live_points=500, verbose=True)

    ### Evaluating the fitted model
    # juliet best fit model
    model = results_full.lc.evaluate(instrument)
    # juliet best fit gp model
    gp_model = results_full.lc.model[instrument]['GP']
    # juliet best fit transit model and its errors
    transit_model = results_full.lc.model[instrument]['deterministic']
    transit_model_err = results_full.lc.model[instrument]['deterministic_errors']

    # Saving the decorrelation plot, if asked
    if save:
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(param[instrument], (fl[instrument]-transit_model), yerr=fle[instrument], fmt='.', alpha=0.3)
        ax1.plot(param[instrument], gp_model, c='k', zorder=100)
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
        plt.savefig(out_path + '/juliet/juliet_full_' + nm_param + '/decorr_' + nm_param +'.png')
        plt.close(fig)
    
    ## Sorting again in time ascending order to make lightcurve plots
    tt3 = Table()
    tt3['tim'] = tim[instrument]
    tt3['fl'] = fl[instrument]
    tt3['fle'] = fle[instrument]
    tt3['model'] = model
    tt3['gp_mod'] = gp_model
    tt3['tran_mod'] = transit_model
    tt3['tran_mod_err'] = transit_model_err

    tt3.sort('tim')

    tim[instrument] = tt3['tim']
    fl[instrument] = tt3['fl']
    fle[instrument] = tt3['fle']
    model = tt3['model']
    gp_model = tt3['gp_mod']
    transit_model = tt3['tran_mod']
    transit_model_err = tt3['tran_mod_err']

    ## Making lightcurves
    if save:
        # Full model
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
        ax1.plot(tim[instrument], model, c='k', zorder=100)
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

        plt.savefig(out_path + '/juliet/juliet_full_' + nm_param + '/full_model_' + nm_param + '.png')
        plt.close(fig)

        # Only transit model
        fac = 1/np.max(transit_model)#1/(1+np.median(mflux))
        # Errors in the model
        umodel, lmodel = transit_model + transit_model_err, transit_model - transit_model_err

        # Making a plot
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instrument], (fl[instrument]-gp_model)*fac, yerr=fle[instrument], fmt='.', alpha=0.3)
        ax1.plot(tim[instrument], transit_model*fac, c='k', zorder=100)
        ax1.fill_between(tim[instrument], umodel*fac, lmodel*fac, color='red', alpha=0.7, zorder=5)
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim[instrument], (fl[instrument]-gp_model-transit_model)*1e6*fac, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--')
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))

        plt.savefig(out_path + '/juliet/juliet_full_' + nm_param + '/transit_model_' + nm_param + '.png')
        plt.close(fig)

    ## Decorrelating!!
    if save:
        tim1, fl1, fle1 = tim[instrument], (fl[instrument]-gp_model)*fac, fle[instrument]
        f1 = open(out_path + '/juliet/juliet_full_' + nm_param + '/' + nm_param + '_decorrelated_photometry.dat','w')
        for i in range(len(tim[instrument])):
            f1.write(str(tim1[i]) + '\t' + str(fl1[i]) + '\t' + str(fle1[i]) + '\n')
        f1.close()

    return results_full.posteriors['lnZ']


#single_param_decorr(tim, fl, fle, param, plan_params, t14, out_path=os.getcwd(), verbose=True, plots=True)

def multiple_params_decorr(tim, fl, fle, params, plan_params, t14, out_path=os.getcwd(), verbose=True):
    """
    This function do the transit light curve analysis with
    lightcurve decorrelation done against a given set of parameters.
    For a single visit
    (Essentially the same as single_param_decorr()) but
    this can take multiple decorrelation vectors.
    ------------------------------------------------------------------
    Parameters:
    -----------
    tim, fl, fle : dict, dict, dict
        transit lightcurve data; time, flux and error in flux
    param : dict
        decorrelation parameter
    plan_params : dict
        juliet readable priors
        juliet will identify which model (batman or catwoman)
        is to be used
    t14 : float
        transit duration
    out_path : str
        output path of the analysed files
        note that everything will be saved in different folders
        which would be sub-folders of a folder called juliet
        default is the present working directory
    verbose : bool
        boolean on whether to print progress of analysis
        default is true
    plots : bool
        boolean on whether to produce various plots for analysis
        This contains: 1) trend in flux with the decorrelation parameter
        and the best fitted GP model to it. 2) Full model fitted 3) best 
        fitted transit model.
    -----------
    return:
    -----------
    .dat file :
        decorrelated photometry stored in out_folder/juliet/juliet_full_param
    lnZ : float
        log Bayesian evidence for the analysis
    """
    ## Instrument
    instrument = list(tim.keys())[0]
    ### Folder to save results
    p1 = Path(out_path + '/FINAL_ANALYSIS_' + instrument)
    if not p1.exists():
        os.mkdir(p1)
    ## Decorrelation vectors
    nms_decorr = list(params.keys())
    lnZ = 0.
    last_used_param = nms_decorr[0]
    params_used = []
    discarded_params = []
    for i in range(len(nms_decorr)):
        nm_decor = nms_decorr[i]
        par_decor1 = params[nm_decor]
        par_decor = {}
        par_decor[nm_decor] = par_decor1
        if len(params_used) != 0:
            tim3, fl3, fle3 = np.loadtxt(out_path + '/juliet/juliet_full_' + last_used_param + '/' + last_used_param + '_decorrelated_photometry.dat',\
                usecols=(0,1,2), unpack=True)
            tim4, fl4, fle4 = {}, {}, {}
            tim4[instrument], fl4[instrument], fle4[instrument] = tim3, fl3, fle3
            ln_z = single_param_decorr(tim=tim4, fl=fl4, fle=fle4, param=par_decor,\
                plan_params=plan_params, t14=t14, out_path=out_path, verbose=verbose, save=False)
        else:
            ln_z = single_param_decorr(tim=tim, fl=fl, fle=fle, param=par_decor,\
                plan_params=plan_params, t14=t14, out_path=out_path, verbose=verbose, save=False)
        print('-----------------------------')
        print('The instrument is: ', instrument)
        print('The last ln(Z) was (for the parameter ' + last_used_param + '): {:.4f}'.format(lnZ))
        print('The new ln(Z) is: ', ln_z)
        xx = input('Do you want to use this analysis further (Y/n)?: ')
        if xx == 'Y' or xx == 'y':
            lnZ = ln_z
            last_used_param = nm_decor
            params_used.append(nm_decor)
            if len(params_used) == 0:
                tim3, fl3, fle3 = np.loadtxt(out_path + '/juliet/juliet_full_' + last_used_param + '/' + last_used_param + '_decorrelated_photometry.dat',\
                    usecols=(0,1,2), unpack=True)
                tim4, fl4, fle4 = {}, {}, {}
                tim4[instrument], fl4[instrument], fle4[instrument] = tim3, fl3, fle3
                ln_z = single_param_decorr(tim=tim4, fl=fl4, fle=fle4, param=par_decor,\
                    plan_params=plan_params, t14=t14, out_path=out_path, verbose=verbose, save=True)
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/decorr_' + nm_decor + '.png ' + p1 + '/decorr_' + nm_decor + '.png')
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/full_model_' + nm_decor + '.png ' + p1 + '/full_model_' + nm_decor + '.png')
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/transit_model_' + nm_decor + '.png ' + p1 + '/transit_model_' + nm_decor + '.png')
            else:
                ln_z = single_param_decorr(tim=tim, fl=fl, fle=fle, param=par_decor,\
                    plan_params=plan_params, t14=t14, out_path=out_path, verbose=verbose, save=True)
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/decorr_' + nm_decor + '.png ' + p1 + '/decorr_' + nm_decor + '.png')
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/full_model_' + nm_decor + '.png ' + p1 + '/full_model_' + nm_decor + '.png')
                os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/transit_model_' + nm_decor + '.png ' + p1 + '/transit_model_' + nm_decor + '.png')
        else:
            discarded_params.append(nm_decor)
    os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/* ' + out_path + '/FINAL_ANALYSIS_' + instrument)
    print(' ')
    print('---------------------------------')
    print(' ')
    print('     Summary of the Analysis     ')
    print(' ')
    print('---------------------------------')
    print('1) Instrument used in analysis: ', instrument)
    print('2) Inngested decorrelation parameters: ')
    print(nms_decorr)
    print('3) Parameter used in decorrelation:')
    print(params_used)
    print('4) Discarded parameters:')
    print(discarded_params)
    print('5) ln(Z) achieved in the analysis: ', lnZ)
    print('6) Final analysis was saved in the folder: ')
    print('   FINAL_ANALYSIS_' + instrument + ' folder in out_path.')

"""
import pycheops
# Downloading data
dd = pycheops.Dataset('CH_PR300024_TG000101_V0200')
tim, fl, fle = dd.get_lightcurve(aperture='OPTIMAL', decontaminate=True)#, reject_highpoints=True)
# To clip outliers (I would, in general, not prefer using this)
tim, fl, fle = dd.clip_outliers(verbose=True);
P, T0, W = 4.7362050, 0.5, 0.2974
tim, fl, fle = dd.flatten(T0, W)

tims, flx, flxe, params = {}, {}, {}, {}
tims['CHEOPS'], flx['CHEOPS'], flxe['CHEOPS'] = tim, fl, fle
params['ROLL'], params['CENX'], params['CENY'], params['BG'] = dd.lc['roll_angle'], dd.lc['centroid_x'], dd.lc['centroid_y'], dd.lc['bg']
params['TIME'] = tim

params_P = ['P_p1', 't0_p1', 'r1_p1', 'r2_p1', 'q1_CHEOPS', 'q2_CHEOPS', 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'loguniform']
hyper_P = [P, [0.5, 0.1], [0.,1.], [0.,1.], [0.,1.], [0.,1.], 0., 90., [1.,100.]]

plan_param_prior = juliet.utils.generate_priors(params_P, dist_P, hyper_P)

multiple_params_decorr(tim=tims, fl=flx, fle=flxe, params=params, plan_params=plan_param_prior, t14=W, out_path=os.getcwd(), verbose=True)
"""