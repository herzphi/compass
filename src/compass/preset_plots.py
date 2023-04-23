#  Making plots
from proxis.modelling import HelperFunctions, Candidate
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def pm_mag_plot(catalogue, catalogue_name, target_name, host_star, candidates_df, band, sigma_min):
    fig, axs = plt.subplots(2, figsize=(7,6))
    for ax, pm_direction in zip(axs.flat, ['pmra', 'pmdec']):
        #  Plot clipped/unclipped data
        ax.plot(
            catalogue[band], 
            catalogue[pm_direction+'_mean'], 
            'o',
            alpha=.5,
            ms=5,
            label=rf'Mean $\mu_{{\alpha, \delta}}$ (50 objects/dot)',
        )
        xline = np.linspace(7,24,100)
        y_option = 'mean'
        column = f'{pm_direction}_{y_option}_model_coeff_{catalogue_name}'
        host_star_data = host_star.__getattribute__(column)
        yaxis = host_star_data[0]*xline+host_star_data[1]
        y_option = 'stddev'
        column = f'{pm_direction}_{y_option}_model_coeff_{catalogue_name}'
        host_star_data_err = host_star.__getattribute__(column)
        if len(host_star_data_err)==3:
            if host_star_data_err[2]<sigma_min:
                intercept = sigma_min
            else:
                intercept = host_star_data_err[2]
            yaxis_error = host_star_data_err[0]*np.exp(-host_star_data_err[1]*xline)+intercept
            yaxis_error[yaxis_error<sigma_min]=sigma_min
        else:
            if host_star_data_err[1]<sigma_min:
                intercept = sigma_min
            else:
                intercept = host_star_data_err[1]
            yaxis_error = host_star_data_err[0]*xline+intercept
            yaxis_error[yaxis_error<sigma_min]=sigma_min
        ax.plot(
            xline, 
            yaxis,
            color='C0',
            label='10th-90th percentile LinFit'
        )
        #  Fit for error
        ax.fill_between(
            xline, 
            yaxis+yaxis_error, 
            yaxis-yaxis_error,
            alpha=.2, 
            color='C3', 
            label='Stddev Bin Fit'
        )
        # Candidate proper motion
        ax.errorbar(
            candidates_df[band], 
            candidates_df[f'{pm_direction}_abs'], 
            yerr=candidates_df[f'{pm_direction}_error'], 
            fmt='.', 
            capsize=1, 
            elinewidth=1, 
            alpha=.2, 
            label='BEAST',
            color='C1'
        )
        ax.set_ylim(catalogue[pm_direction+'_mean'].min()-25,catalogue[pm_direction+'_mean'].max()+25)
    #axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel(r'$\mu_{ra}$ [mas/yr]')
    axs[1].set_ylabel(r'$\mu_{dec}$ [mas/yr]')
    axs[0].set_xlabel('K-Band [mag]')
    axs[1].set_xlabel('K-Band [mag]')

    # Adding a spline based on Gaia and the survey data weighted by the pm error
    fig.suptitle(f'Gaussian2D fitting parameters {target_name}', fontsize=15)
    plt.tight_layout()


def p_ratio_plot(candidate_object, target, band):
    #  Contour plots
    fig, axs = plt.subplots(2, 2, figsize=(7,7))

    #  Find a suitable range for the mgrid. 
    mg_list = [
        abs(candidate_object.g2d_conv.x_mean+6*candidate_object.g2d_conv.x_stddev), abs(candidate_object.g2d_conv.y_mean+6*candidate_object.g2d_conv.y_stddev), \
            abs(candidate_object.g2d_cc.x_mean+6*np.sqrt(candidate_object.g2d_cc.x_stddev**2)), abs(candidate_object.g2d_cc.y_mean+6*np.sqrt(candidate_object.g2d_cc.y_stddev**2))
    ]
    mg = max(mg_list)
    step = 100
    y, x = np.mgrid[-mg:mg:100j, -mg:mg:100j]
    xline = np.linspace(-mg, mg, 400)
    for confd in [.5, .9, .99]:
        major_axis, minor_axis, angle = HelperFunctions.get_ellipse_props(candidate_object.cov_conv, confidence=confd)
        HelperFunctions.add_ellp_patch(candidate_object.g2d_conv, major_axis, minor_axis, angle, 'C0', axs[1,0])

    for confd in [.5, .9, .99]:
        major_axis, minor_axis, angle = HelperFunctions.get_ellipse_props(candidate_object.cov_pmuM1, confidence=confd)
        HelperFunctions.add_ellp_patch(candidate_object.g2d_pmuM1, major_axis, minor_axis, angle, 'C1', axs[1,0])

    axs[1,0].plot(candidate_object.g2d_cc.x_mean, candidate_object.g2d_cc.y_mean, color='red', marker='x', label='candidate')
    #  mu_ra plot
    axs[0,0].plot(xline, HelperFunctions.gaussian1D(candidate_object.g2d_conv, 'x')(xline), label=r'$p(\mu_{\alpha}|M_b)$')
    axs[0,0].plot(xline, HelperFunctions.gaussian1D(candidate_object.g2d_pmuM1, 'x')(xline), label=r'$p(\mu_{\alpha}|M_{tc})$')
    axs[0,0].set_ylabel('Probability')
    #  mu_dec plot
    axs[1,1].plot(HelperFunctions.gaussian1D(candidate_object.g2d_conv, 'y')(xline), xline, label=r'$p(\mu_{\delta}|M_b)$')
    axs[1,1].plot(HelperFunctions.gaussian1D(candidate_object.g2d_pmuM1, 'y')(xline), xline, label=r'$p(\mu_{\delta}|M_{tc})$')
    axs[1,1].set_xlabel('Probability')

    max_yaxis = max([*HelperFunctions.gaussian1D(candidate_object.g2d_conv, 'x')(xline), *HelperFunctions.gaussian1D(candidate_object.g2d_pmuM1, 'x')(xline)])
    max_xaxis = max([*HelperFunctions.gaussian1D(candidate_object.g2d_conv, 'y')(xline), *HelperFunctions.gaussian1D(candidate_object.g2d_pmuM1, 'y')(xline)])
    textyaxis = max_yaxis
    textyaxis_step = textyaxis/8
    #  Add lines to indicate candidate_objects position
    axs[0,0].vlines(candidate_object.g2d_cc.x_mean, 0, textyaxis, color='red', linestyles='dashed', label=r'$\mu_{true}$')
    axs[1,1].hlines(candidate_object.g2d_cc.y_mean, 0, max_xaxis, color='red', linestyles='dashed', label=r'$\mu_{true}$')
    #  Add text
    axs[0,1].text(0, 1-6/6, \
        r'$\log\frac{{p(\mu_{{\alpha, \delta}}|M_{tc})}}{{p(\mu_{{\alpha,\delta}}|M_b)}}$'+f' = {candidate_object.p_ratio:.2f}')
    axs[0,1].text(0, 1-5/6, r'$p(\mu_{{\alpha,\delta}}|M_b)$'+' = %.2E' % Decimal(candidate_object.p_b))
    axs[0,1].text(0, 1-4/6, r'$p(\mu_{{\alpha,\delta}}|M_{tc})$'+' = %.2E' % Decimal(candidate_object.p_tc))
    axs[0,1].text(0, 1-3/6, r'$\mu_{\alpha, \delta}$'+f' = [{candidate_object.g2d_cc.x_mean.value:.2f}, {candidate_object.g2d_cc.y_mean.value:.2f}]')
    axs[0,1].text(0, 1-2/6, 'candidates proper motion [mas/yr]:')
    axs[0,1].text(0, 1-1/6, rf'$\sigma^{{M_b}}_{{pmra}}$={np.sqrt(candidate_object.cov_conv[0,0]):.0f}, $\sigma^{{M_b}}_{{pmdec}}$={np.sqrt(candidate_object.cov_conv[1,1]):.0f}')

    #  Set labels and legend
    axs[1,0].set_xlabel(r'$\mu_{\alpha} $ mas/yr')
    axs[1,0].set_ylabel(r'$\mu_{\delta} $ mas/yr')
    fig.suptitle(f'Candidate is a {candidate_object.back_true} \n {target} candidate at {candidate_object.cc_true_data[band]:.2f} $mag$, final_uuid={candidate_object.final_uuid}')
    #axs[0,1].remove()
    axs[0,1].set_axis_off()
    axs[0,0].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    #  Set all the same axis limits
    #p_lim = 0.01
    mg = np.max([candidate_object.g2d_conv.x_mean.value+5*np.sqrt(candidate_object.cov_conv[0,0]), 5*np.sqrt(candidate_object.cov_pmuM1[0,0])])
    axs[1,0].set_xlim(-mg,mg)
    axs[0,0].set_xlim(-mg,mg)
    mg = np.max([candidate_object.g2d_conv.y_mean.value+5*np.sqrt(candidate_object.cov_conv[1,1]), 5*np.sqrt(candidate_object.cov_pmuM1[1,1])])
    axs[1,1].set_ylim(-mg,mg)


def odds_ratio_sep_mag_plot(candidates_table, target_name):
    fig, axs = plt.subplots(1, figsize=(7,3))
    g = sns.scatterplot(
        x='sep_mean',
        y='p_ratio',
        data=candidates_table,
        hue='band',
        style='p_ratio_catalogue',
        s=60,
        ax=axs
    )
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs.hlines(0, 1700, 5000, linestyles='dotted', colors='gray')
    axs.set_ylim(-50, 30)

    axs.set_ylabel('Odds Ratio: '+r'$\log_{10}\frac{{P(\mu|M_{tc})}}{{P(\mu|M_b)}}$')
    axs.set_xlabel(r'Separation $[mas]$')
    axs.set_title(f'Odds ratios of all candidates of {target_name}')
    plt.tight_layout()
    plt.show()