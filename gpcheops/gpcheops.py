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


def regress(x):
    """
    To normalise x
    (For usage in normalise decorrelation parameter)
    """
    return (x - np.mean(x))/np.sqrt(np.var(x))


def single_param_decorr(tim, fl, fle, param, plan_params, t14, GP='ExM', out_path=os.getcwd(), sampler='dynesty', save=True, oot_method='single', verbose=True):
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
    GP : str
        On which GP kernel to use. ExM for Exponential-Matern kernel
        QP for quasi-periodic kernel, SHO for Simple Harmonic Oscillator kernel
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
        raise Exception('Method to discard out-of-transit/eclipse points can only be "sinle" or "multi".')
    
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
    if GP == 'ExM':
        params_gp = ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform']
        hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
    elif GP == 'QP':
        params_gp = ['GP_B_' + instrument, 'GP_C_' + instrument, 'GP_L_' + instrument, 'GP_Prot_' + instrument]
        dist_gp = ['loguniform', 'loguniform', 'loguniform','loguniform']
        hyper_gp = [[1e-5,1e4], [1e-5,1e4], [1e-5, 1e4], [1.,1e2]]
    elif GP == 'SHO':
        params_gp = ['GP_S0_' + instrument, 'GP_omega0_' + instrument, 'GP_Q_' + instrument]
        dist_gp = ['uniform', 'uniform', 'fixed']
        hyper_gp = [[np.exp(-40.), np.exp(0.)], [np.exp(-10.), np.exp(10.)], np.exp(1/np.sqrt(2))]
    else:
        raise Exception('GP method can only be ExM, QP or SHO.')
    # Total priors
    params_gp_only = params_ins + params_gp
    dist_gp_only = dist_ins + dist_gp
    hyper_gp_only = hyper_ins + hyper_gp
    # Populating prior dict
    priors = juliet.utils.generate_priors(params_gp_only, dist_gp_only, hyper_gp_only)

    ## Running GP only fit
    data = juliet.load(priors=priors, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, GP_regressors_lc=param_oot,\
         out_folder=out_path + '/juliet_'+ instrument +'/juliet_oot_' + nm_param)
    res_gp_only = data.fit(sampler = sampler, n_live_points=500, verbose = verbose)

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
        if dist_gp[i] != 'fixed':
            post1 = res_gp_only.posteriors['posterior_samples'][params_gp[i]]
            mu, sig = np.median(post1), np.std(post1)
            dist_gp[i] = 'normal'
            hyper_gp[i] = [mu, 2*sig]#, hyper_gp[i][0], hyper_gp[i][1]]
    # Same goes for mflux and sigma_w
    # For sigma_w_CHEOPS
    dist_ins[2] = 'normal'
    post2 = res_gp_only.posteriors['posterior_samples']['sigma_w_' + instrument]
    mu, sig = np.median(post2), np.std(post2)
    hyper_ins[2] = [mu, 2*sig]#, hyper_ins[2][0], hyper_ins[2][1]]
    # For mflux
    dist_ins[1] = 'normal'
    post2 = res_gp_only.posteriors['posterior_samples']['mflux_' + instrument]
    mu, sig = np.median(post2), np.std(post2)
    hyper_ins[1] = [mu, 2*sig]
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
         out_folder=out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param)
    results_full = data_full.fit(sampler = sampler, n_live_points=500, verbose=True)

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
        plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/decorr_' + nm_param +'.png')
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

        plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/full_model_' + nm_param + '.png')
        plt.close(fig)

        # Only transit model
        fac = 1/np.max(transit_model)#1/(1+np.median(mflux))
        # Errors in the model
        umodel, lmodel = transit_model + transit_model_err, transit_model - transit_model_err

        # Making a plot
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        ## 
        dummy_tim, dummy_param = {}, {}
        tt4 = Table()
        tt4['time'], tt4['param'] = tim[instrument], param[instrument]
        tt4.sort('param')
        dummy_tim[instrument], dummy_param[instrument] = tt4['time'], tt4['param']

        t2 = np.linspace(np.min(dummy_tim[instrument]), np.max(dummy_tim[instrument]), 10000)
        #gp2 = np.linspace(np.min(dummy_param[instrument]), np.max(dummy_param[instrument]), 1000)
        cs = CubicSpline(dummy_tim[instrument], dummy_param[instrument])
        gp2 = cs(t2)
        model_res = results_full.lc.evaluate(instrument, t=t2, GPregressors=gp2)
        trans_model = results_full.lc.model[instrument]['deterministic']

        tt5 = Table()
        tt5['time'], tt5['param'], tt5['transit_model'] = t2, gp2, trans_model
        tt5.sort('time')
        t2, gp2, trans_model = tt5['time'], tt5['param'], tt5['transit_model']
        fac1 = 1/np.max(trans_model)

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instrument], (fl[instrument]-gp_model)*fac, yerr=fle[instrument], fmt='.', alpha=0.3)
        #ax1.plot(tim[instrument], transit_model*fac, c='k', zorder=100)
        ax1.plot(t2, trans_model*fac1, c='k', zorder=100)
        #ax1.fill_between(tim[instrument], umodel*fac, lmodel*fac, color='red', alpha=0.7, zorder=5)
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

        if transit:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/transit_model_' + nm_param + '.png')
        elif eclipse:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/eclipse_model_' + nm_param + '.png')
        else:
            plt.savefig(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/transit_eclipse_model_' + nm_param + '.png')
        plt.close(fig)

    ## Decorrelating!!
    if save:
        tim1, fl1, fle1 = tim[instrument], (fl[instrument]-gp_model)*fac, fle[instrument]
        f1 = open(out_path + '/juliet_'+ instrument +'/juliet_full_' + nm_param + '/' + nm_param + '_decorrelated_photometry.dat','w')
        for i in range(len(tim[instrument])):
            f1.write(str(tim1[i]) + '\t' + str(fl1[i]) + '\t' + str(fle1[i]) + '\n')
        f1.close()

    return results_full.posteriors['lnZ']


#single_param_decorr(tim, fl, fle, param, plan_params, t14, out_path=os.getcwd(), verbose=True, plots=True)

def multiple_params_decorr(tim, fl, fle, params, plan_params, t14, GP='ExM', sampler='dynesty', out_path=os.getcwd(), oot_method='single', verbose=True):
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
    GP : str
        On which GP kernel to use. ExM for Exponential-Matern kernel
        QP for quasi-periodic kernel, SHO for Simple Harmonic Oscillator kernel
        Default is ExM
    sampler : str
        sampler to be used in posterior estimation
        a valid choices are 'multinest', 'dynesty', 'dynamic_dynesty', or 'ultranest'
        default is 'dynesty'
    out_path : str
        output path of the analysed files
        note that everything will be saved in different folders
        which would be sub-folders of a folder called juliet
        default is the present working directory
    oot_method : str
        Which method to be use for selecting out-of-transit/eclipse (or joint) points
        default is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
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
    # Determining what kind of fitting is it
    eclipse = False
    transit = False
    for j in plan_params.keys():
        if j[0:2] == 'fp':
            eclipse = True
        if j[0:2] == 'q1':
            transit = True
    ## Instrument
    instrument = list(tim.keys())[0]
    ### Folder to save results
    p1 = Path(out_path + '/FINAL_ANALYSIS_' + instrument)
    if not p1.exists():
        os.mkdir(p1)
    ## Decorrelation vectors
    nms_decorr = list(params.keys())
    lnZ = 0.
    last_used_param = 'NONE'
    params_used = []
    discarded_params = []
    for i in range(len(nms_decorr)):
        nm_decor = nms_decorr[i]
        par_decor1 = params[nm_decor]
        par_decor = {}
        par_decor[nm_decor] = par_decor1
        if len(params_used) != 0:
            tim3, fl3, fle3 = np.loadtxt(out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/' + last_used_param + '_decorrelated_photometry.dat',\
                usecols=(0,1,2), unpack=True)
            tim4, fl4, fle4 = {}, {}, {}
            tim4[instrument], fl4[instrument], fle4[instrument] = tim3, fl3, fle3
            ln_z = single_param_decorr(tim=tim4, fl=fl4, fle=fle4, param=par_decor,\
                plan_params=plan_params, t14=t14, GP=GP, out_path=out_path, sampler=sampler, verbose=verbose, oot_method=oot_method, save=False)
        else:
            ln_z = single_param_decorr(tim=tim, fl=fl, fle=fle, param=par_decor,\
                plan_params=plan_params, t14=t14, GP=GP, out_path=out_path, sampler=sampler, verbose=verbose, oot_method=oot_method, save=False)
        print('-----------------------------')
        print('The instrument is: ', instrument)
        print('The last ln(Z) was (for the parameter ' + last_used_param + '): {:.4f}'.format(lnZ))
        print('The new ln(Z) is (for the new parameter ' + nm_decor + '): ', ln_z)
        xx = input('Do you want to use this analysis further (Y/n)?: ')
        if xx == 'Y' or xx == 'y':
            if len(params_used) != 0:
                tim3, fl3, fle3 = np.loadtxt(out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/' + last_used_param + '_decorrelated_photometry.dat',\
                    usecols=(0,1,2), unpack=True)
                tim4, fl4, fle4 = {}, {}, {}
                tim4[instrument], fl4[instrument], fle4[instrument] = tim3, fl3, fle3
                ln_z = single_param_decorr(tim=tim4, fl=fl4, fle=fle4, param=par_decor,\
                    plan_params=plan_params, t14=t14, GP=GP, out_path=out_path, sampler=sampler, verbose=verbose, oot_method=oot_method, save=True)
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/decorr_' + nm_decor + '.png ' + p1 + '/decorr_' + nm_decor + '.png')
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/full_model_' + nm_decor + '.png ' + p1 + '/full_model_' + nm_decor + '.png')
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/transit_model_' + nm_decor + '.png ' + p1 + '/transit_model_' + nm_decor + '.png')
            else:
                ln_z = single_param_decorr(tim=tim, fl=fl, fle=fle, param=par_decor,\
                    plan_params=plan_params, t14=t14, GP=GP, out_path=out_path, sampler=sampler, verbose=verbose, oot_method=oot_method, save=True)
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/decorr_' + nm_decor + '.png ' + p1 + '/decorr_' + nm_decor + '.png')
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/full_model_' + nm_decor + '.png ' + p1 + '/full_model_' + nm_decor + '.png')
                #os.system('cp ' + out_path + '/juliet/juliet_full_' + last_used_param + '/transit_model_' + nm_decor + '.png ' + p1 + '/transit_model_' + nm_decor + '.png')
            lnZ = ln_z
            last_used_param = nm_decor
            params_used.append(nm_decor)
            os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/decorr_' + nm_decor + '.png ' + p1 + '/decorr_' + nm_decor + '.png')
            os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/full_model_' + nm_decor + '.png ' + p1 + '/full_model_' + nm_decor + '.png')
            if transit:
                os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/transit_model_' + nm_decor + '.png ' + p1 + '/transit_model_' + nm_decor + '.png')
            elif eclipse:
                os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/eclipse_model_' + nm_decor + '.png ' + p1 + '/eclipse_model_' + nm_decor + '.png')
            elif transit and eclipse:
                os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/transit_eclipse_model_' + nm_decor + '.png ' + p1 + '/transit_eclipse_model_' + nm_decor + '.png')
        else:
            discarded_params.append(nm_decor)
    os.system('cp ' + out_path + '/juliet_'+ instrument +'/juliet_full_' + last_used_param + '/* ' + out_path + '/FINAL_ANALYSIS_' + instrument)
    print(' ')
    print('---------------------------------')
    print(' ')
    print('     Summary of the Analysis     ')
    print(' ')
    print('---------------------------------')
    print('1) Instrument used in analysis: ', instrument)
    print('2) Ingested decorrelation parameters: ')
    print(nms_decorr)
    print('3) Parameter used in decorrelation:')
    print(params_used)
    print('4) Discarded parameters:')
    print(discarded_params)
    print('5) ln(Z) achieved in the analysis: ', lnZ)
    print('6) Final analysis was saved in the folder: ')
    print('   FINAL_ANALYSIS_' + instrument + ' folder in out_path.')


def multiple_visits(input_folders, plan_params, t14, oot_method, out_path=os.getcwd(), GP='ExM', jointGP=False, sampler='dynesty', verbose=True):
    """
    This function will analyse multiple visits analysed
    by multiple_params_decorr function
    ---------------------------------------------------
    Parameters:
    -----------
    input_folders : list
        list containing folders analysed by multiple_params_decorr
    plan_params : dict
        juliet readable planetary priors
    t14 : float
        transit duration in days
    oot_method : list
        Which method to be use for selecting out-of-transit/eclipse (or joint) points for each folder
        one option is single-- this method takes transit duration and and discard points
                            out of this time. Best to use for single transit/eclipse.
        another option is multi -- which uses phase-space to discard out-of-transit/eclipse
                                   points. Can be used with single/multiple transit/eclipse events.
    out_path : str
        path to the output folder
        default is the present working directory
    GP : str
        On which GP kernel to use. ExM for Exponential-Matern kernel
        QP for quasi-periodic kernel, SHO for Simple Harmonic Oscillator kernel
        Default is ExM
    jointGP : bool
        boolean on whether to provide a joint GP priors on each intruments
        default is False
    sampler : str
        sampler to be used in posterior estimation
        a valid choices are 'multinest', 'dynesty', 'dynamic_dynesty', or 'ultranest'
        default is 'dynesty'
    verbose : bool
        boolean on whether to print progress of analysis
        default is true
    """
    # Creating a dictionary to save data products
    tim, fl, fle = {}, {}, {}
    tim_oot, fl_oot, fle_oot = {}, {}, {}
    # GP priors
    par_gp, dist_gp, hyper_gp = [], [], []
    # instrumental priors
    par_ins, dist_ins, hyper_ins = [], [], []
    # Saving the instruments
    if len(input_folders) != len(oot_method):
        raise Exception('Not enough `oot-method` was provided for total input folders, or\n Not enough folders for `oot-method`.')
    instruments = []
    for i in range(len(input_folders)):
        instrument = input_folders[i].split('_')[-1]
        instruments.append(instrument)
        # Saving decorrelated data
        data_lc = glob(input_folders[i] + '/*_decorrelated_photometry.dat')[0]
        tim_lc, fl_lc, fle_lc = np.loadtxt(data_lc, usecols=(0,1,2), unpack=True)
        tim[instrument], fl[instrument], fle[instrument] = tim_lc, fl_lc, fle_lc
        # Saving priors
        pc1 = glob(input_folders[i] + '/*.pkl')[0]
        print(pc1)
        post1 = pickle.load(open(pc1, 'rb'), encoding='latin1')
        pp1 = post1['posterior_samples']
        # GP priors
        """
        for j in pp1.keys():
            if j[0:2] == 'GP':
                par_gp.append(j)
                dist_gp.append('normal')
                mu, sig = np.median(pp1[j]), 2*np.std(pp1[j])
                hyper_gp.append([mu, sig])
        """
        # GP parameters
        if not jointGP:
            if GP == 'ExM':
                par_gp = par_gp + ['GP_sigma_' + instrument, 'GP_timescale_' + instrument, 'GP_rho_' + instrument]
                dist_gp = dist_gp + ['loguniform', 'loguniform', 'loguniform']
                hyper_gp = hyper_gp + [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
            elif GP == 'QP':
                par_gp = par_gp + ['GP_B_' + instrument, 'GP_C_' + instrument, 'GP_L_' + instrument, 'GP_Prot_' + instrument]
                dist_gp = dist_gp + ['loguniform', 'loguniform', 'loguniform','loguniform']
                hyper_gp = hyper_gp + [[1e-5,1e4], [1e-5,1e4], [1e-5, 1e4], [1.,1e2]]
            elif GP == 'SHO':
                par_gp = par_gp + ['GP_S0_' + instrument, 'GP_omega0_' + instrument, 'GP_Q_' + instrument]
                dist_gp = dist_gp + ['uniform', 'uniform', 'fixed']
                hyper_gp = hyper_gp + [[np.exp(-40.), np.exp(0.)], [np.exp(-10.), np.exp(10.)], np.exp(1/np.sqrt(2))]
        # instrumental priors
        # mdilution
        par_ins.append('mdilution_' + instrument)
        dist_ins.append('fixed')
        hyper_ins.append(1.0)
        # mflux and sigma_w
        for j in pp1.keys():
            if j[0:5] == 'mflux' or j[0:7] == 'sigma_w':
                par_ins.append(j)
                dist_ins.append('normal')
                mu, sig = np.median(pp1[j]), np.std(pp1[j])
                hyper_ins.append([mu, sig])
        t01 = np.median(pp1['t0_p1'])
        try:
            p01 = np.median(pp1['P_p1'])
        except:
            p01 = plan_params['P_p1']['hyperparameters']
        eclipse = False
        transit = False
        for k in pp1.keys():
            if k[0:2] == 'fp':
                eclipse = True
            if k[0:2] == 'q1':
                transit = True
        if oot_method == oot_method[i]:
            if transit and not eclipse:
                mask = np.where(tim_lc > (t01 + (t14/2)))[0]
                mask = np.hstack((np.where(tim_lc < (t01 - (t14/2)))[0], mask))
            if eclipse and not transit:
                mask = np.where(tim_lc > (t01 + (p01/2) + (t14/2)))[0]
                mask = np.hstack((np.where(tim_lc < (t01 + (p01/2) - (t14/2)))[0], mask))
            if eclipse and transit:
                mask = np.where(tim_lc > (t01 + (t14/2)))[0]
                mask = np.hstack((np.where(tim_lc < (t01 - (t14/2)))[0], mask))
                mask = np.hstack((np.where(tim_lc < (t01 + (p01/2) + (t14/2)))[0], mask))
                mask = np.hstack((np.where(tim_lc < (t01 + (p01/2) - (t14/2)))[0], mask))
        elif oot_method == oot_method[i]:
            phs_t = juliet.utils.get_phases(tim, p01, t01)
            phs_e = juliet.utils.get_phases(tim, p01, (t01+(p01/2)))
            if eclipse and not transit:
                mask = np.where(np.abs(phs_e*p01) >= t14)[0]
            elif transit and not eclipse:
                mask = np.where(np.abs(phs_t*p01) >= t14)[0]
            elif transit and eclipse:
                mask = np.where((np.abs(phs_e*p01) >= t14)&(np.abs(phs_t*p01) >= t14))[0]
        else:
            raise Exception('Method to discard out-of-transit/eclipse points can only be "sinle" or "multi".')
        tim_lc2, fl_lc2, fle_lc2 = tim_lc[mask], fl_lc[mask], fle_lc[mask]
        tim_oot[instrument], fl_oot[instrument], fle_oot[instrument] = tim_lc2, fl_lc2, fle_lc2
    # So, now, we have data from multiple instruments and corresponding priors
    # If joint GP priors has to be provided
    if jointGP:
        if GP == 'ExM':
            par1, par2, par3 = 'GP_sigma', 'GP_timescale', 'GP_rho'
            for k in instruments:
                par1, par2, par3 = par1 + '_' + k, par2 + '_' + k, par3 + '_' + k
            par_gp = [par1, par2, par3]
            dist_gp = ['loguniform', 'loguniform', 'loguniform']
            hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]
        elif GP == 'QP':
            par1, par2, par3, par4 = 'GP_B', 'GP_C', 'GP_L', 'GP_Prot'
            for k in instruments:
                par1, par2, par3, par4 = par1 + '_' + k, par2 + '_' + k, par3 + '_' + k, par4 + '_' + k
            par_gp = [par1, par2, par3, par4]
            dist_gp = ['loguniform', 'loguniform', 'loguniform','loguniform']
            hyper_gp = [[1e-5,1e4], [1e-5,1e4], [1e-5, 1e4], [1.,1e2]]
        elif GP == 'SHO':
            par1, par2, par3 = 'GP_S0', 'GP_omega0', 'GP_Q'
            for k in instruments:
                par1, par2, par3 = par1 + '_' + k, par2 + '_' + k, par3 + '_' + k
            par_gp = [par1, par2, par3]
            dist_gp = ['uniform', 'uniform', 'fixed']
            hyper_gp = [[np.exp(-40.), np.exp(0.)], [np.exp(-10.), np.exp(10.)], np.exp(1/np.sqrt(2))]
        else:
            raise Exception('GP method can only be ExM, QP or SHO.')
    ### Folder to save results
    pth1 = Path(out_path + '/FINAL_ANALYSIS_MULT_INSTRUMENT')
    if not pth1.exists():
        os.mkdir(pth1)
    # We first fit the out of transit data
    # Total priors
    params_gp_only = par_ins + par_gp
    dist_gp_only = dist_ins + dist_gp
    hyper_gp_only = hyper_ins + hyper_gp
    # Populating prior dict
    priors = juliet.utils.generate_priors(params_gp_only, dist_gp_only, hyper_gp_only)

    ## Running GP only fit
    data = juliet.load(priors=priors, t_lc=tim_oot, y_lc=fl_oot, yerr_lc=fle_oot, GP_regressors_lc=tim_oot,\
         out_folder=pth1 + '/oot')
    res_gp_only = data.fit(sampler = sampler, n_live_points=500, verbose = verbose)

    ## Full data fit
    ## Defining priors
    # We would take instrumental priors from our previous fit
    for i in range(len(par_gp)):
        if dist_gp[i] != 'fixed':
            post1 = res_gp_only.posteriors['posterior_samples'][par_gp[i]]
            mu, sig = np.median(post1), np.std(post1)
            dist_gp[i] = 'normal'
            hyper_gp[i] = [mu, sig]#, hyper_gp[i][0], hyper_gp[i][1]]
    # Same goes for mflux and sigma_w
    for i in range(len(par_ins)):
        if dist_ins[i] != 'fixed':
            post1 = res_gp_only.posteriors['posterior_samples'][par_ins[i]]
            mu, sig = np.median(post1), np.std(post1)
            dist_ins[i] = 'normal'
            hyper_ins[i] = [mu, sig]#, hyper_ins[i][0], hyper_ins[i][1]]
    # Planetary parameters
    params_P, dist_P, hyper_P = list(plan_params.keys()), [], []
    for k in plan_params.keys():
        dist_P.append(plan_params[k]['distribution'])
        hyper_P.append(plan_params[k]['hyperparameters'])
    # Total priors
    params = params_P + par_ins + par_gp
    dist = dist_P + dist_ins + dist_gp
    hyper = hyper_P + hyper_ins + hyper_gp
    # Prior dictionary
    priors = juliet.utils.generate_priors(params, dist, hyper)

    # Running the whole fit
    data_full = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, GP_regressors_lc=tim,\
         out_folder=pth1)
    results_full = data_full.fit(sampler = sampler, n_live_points=500, verbose=True)

    for i in range(len(instruments)):
        ### Evaluating the fitted model
        # juliet best fit model
        model = results_full.lc.evaluate(instruments[i])
        # juliet best fit gp model
        gp_model = results_full.lc.model[instruments[i]]['GP']
        # juliet best fit transit model and its errors
        transit_model = results_full.lc.model[instruments[i]]['deterministic']
        transit_model_err = results_full.lc.model[instruments[i]]['deterministic_errors']

        # Plotting the results
        # Full model
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instruments[i]], fl[instruments[i]], yerr=fle[instruments[i]], fmt='.', alpha=0.3)
        ax1.plot(tim[instruments[i]], model, c='k', zorder=100)
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim[instruments[i]]), np.max(tim[instruments[i]]))
        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim[instruments[i]], (fl[instruments[i]]-model)*1e6, yerr=fle[instruments[i]]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--')
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim[instruments[i]]), np.max(tim[instruments[i]]))

        plt.savefig(pth1 + '/full_model_' + instruments[i] + '.png')
        plt.close(fig)

        # Only transit model
        fac = 1/np.max(transit_model)#1/(1+np.median(mflux))
        # Errors in the model
        umodel, lmodel = transit_model + transit_model_err, transit_model - transit_model_err

        # Making a plot
        fig = plt.figure(figsize=(16,9))
        gs = gd.GridSpec(2,1, height_ratios=[2,1])

        t2 = np.linspace(np.min(tim[instruments[i]]), np.max(tim[instruments[i]]), 10000)
        model_res = results_full.lc.evaluate(instruments[i], t=t2, GPregressors=t2)
        trans_model = results_full.lc.model[instruments[i]]['deterministic']
        fac1 = 1/np.max(trans_model)

        # Top panel
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(tim[instruments[i]], (fl[instruments[i]]-gp_model)*fac, yerr=fle[instruments[i]], fmt='.', alpha=0.3)
        #ax1.plot(tim[instruments[i]], transit_model*fac, c='k', zorder=100)
        ax1.plot(t2, trans_model*fac1, c='k', zorder=100)
        #ax1.fill_between(tim[instruments[i]], umodel*fac, lmodel*fac, color='red', alpha=0.7, zorder=5)
        ax1.set_ylabel('Relative Flux')
        ax1.set_xlim(np.min(tim[instruments[i]]), np.max(tim[instruments[i]]))
        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        # Bottom panel
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(tim[instruments[i]], (fl[instruments[i]]-gp_model-transit_model)*1e6*fac, yerr=fle[instruments[i]]*1e6, fmt='.', alpha=0.3)
        ax2.axhline(y=0.0, c='black', ls='--')
        ax2.set_ylabel('Residuals (ppm)')
        ax2.set_xlabel('Time (BJD)')
        ax2.set_xlim(np.min(tim[instruments[i]]), np.max(tim[instruments[i]]))

        if transit:
            plt.savefig(pth1 + '/transit_model_' + instruments[i] + '.png')
        elif eclipse:
            plt.savefig(pth1 + '/eclipse_model_' + instruments[i] + '.png')
        else:
            plt.savefig(pth1 + '/transit_eclipse_model_' + instruments[i] + '.png')
        plt.close(fig)

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