#  Making plots
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from compass import helperfunctions


# Mabye drop pm_mag_plot function
def pm_mag_plot(
    catalogue, catalogue_name, target_name, host_star, candidates_df, band, sigma_min
):
    fig, axs = plt.subplots(2, figsize=(7, 6))
    for ax, pm_direction in zip(axs.flat, ["pmra", "pmdec"]):
        #  Plot clipped/unclipped data
        ax.plot(
            catalogue[band],
            catalogue[pm_direction + "_mean"],
            "o",
            alpha=0.5,
            ms=5,
            label=r"Mean $\mu_{{\alpha, \delta}}$ (50 objects/dot)",
        )
        xline = np.linspace(7, 24, 100)
        y_option = "mean"
        column = f"{pm_direction}_{y_option}_model_coeff_{catalogue_name}"
        host_star_data = host_star.__getattribute__(column)
        yaxis = host_star_data[0] * xline + host_star_data[1]
        y_option = "stddev"
        column = f"{pm_direction}_{y_option}_model_coeff_{catalogue_name}"
        host_star_data_err = host_star.__getattribute__(column)
        if len(host_star_data_err) == 3:
            if host_star_data_err[2] < sigma_min:
                intercept = sigma_min
            else:
                intercept = host_star_data_err[2]
            yaxis_error = (
                host_star_data_err[0] * np.exp(-host_star_data_err[1] * xline)
                + intercept
            )
            yaxis_error[yaxis_error < sigma_min] = sigma_min
        else:
            if host_star_data_err[1] < sigma_min:
                intercept = sigma_min
            else:
                intercept = host_star_data_err[1]
            yaxis_error = host_star_data_err[0] * xline + intercept
            yaxis_error[yaxis_error < sigma_min] = sigma_min
        ax.plot(xline, yaxis, color="C0", label="10th-90th percentile LinFit")
        #  Fit for error
        ax.fill_between(
            xline,
            yaxis + yaxis_error,
            yaxis - yaxis_error,
            alpha=0.2,
            color="C3",
            label="Stddev Bin Fit",
        )
        # Candidate proper motion
        ax.errorbar(
            candidates_df[band],
            candidates_df[f"{pm_direction}_abs"],
            yerr=candidates_df[f"{pm_direction}_error"],
            fmt=".",
            capsize=1,
            elinewidth=1,
            alpha=0.2,
            label="BEAST",
            color="C1",
        )
        ax.set_ylim(
            catalogue[pm_direction + "_mean"].min() - 25,
            catalogue[pm_direction + "_mean"].max() + 25,
        )
    # axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel(r"$\mu_{ra}$ [mas/yr]")
    axs[1].set_ylabel(r"$\mu_{dec}$ [mas/yr]")
    axs[0].set_xlabel("K-Band [mag]")
    axs[1].set_xlabel("K-Band [mag]")

    # Adding a spline based on Gaia and the survey data weighted by the pm error
    fig.suptitle(f"Gaussian2D fitting parameters {target_name}", fontsize=15)
    plt.tight_layout()


def p_ratio_plot(candidate_object, target, band):
    #  Contour plots
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    #  Find a suitable range for the mgrid.
    mg_list = [
        abs(candidate_object.g2d_conv.x_mean + 6 * candidate_object.g2d_conv.x_stddev),
        abs(candidate_object.g2d_conv.y_mean + 6 * candidate_object.g2d_conv.y_stddev),
        abs(
            candidate_object.g2d_cc.x_mean
            + 6 * np.sqrt(candidate_object.g2d_cc.x_stddev**2)
        ),
        abs(
            candidate_object.g2d_cc.y_mean
            + 6 * np.sqrt(candidate_object.g2d_cc.y_stddev**2)
        ),
    ]
    mg = max(mg_list)
    step = 100
    y, x = np.mgrid[-mg:mg:100j, -mg:mg:100j]
    xline = np.linspace(-mg, mg, 400)
    for confd in [0.5, 0.9, 0.99]:
        major_axis, minor_axis, angle = helperfunctions.get_ellipse_props(
            candidate_object.cov_conv, confidence=confd
        )
        helperfunctions.add_ellp_patch(
            candidate_object.g2d_conv, major_axis, minor_axis, angle, "C0", axs[1, 0]
        )

    for confd in [0.5, 0.9, 0.99]:
        major_axis, minor_axis, angle = helperfunctions.get_ellipse_props(
            candidate_object.cov_pmuM1, confidence=confd
        )
        helperfunctions.add_ellp_patch(
            candidate_object.g2d_pmuM1, major_axis, minor_axis, angle, "C1", axs[1, 0]
        )

    axs[1, 0].plot(
        candidate_object.g2d_cc.x_mean,
        candidate_object.g2d_cc.y_mean,
        color="red",
        marker="x",
        label="candidate",
    )
    #  mu_ra plot
    axs[0, 0].plot(
        xline,
        helperfunctions.gaussian1D(candidate_object.g2d_conv, "x")(xline),
        label=r"$p(\mu_{\alpha}|M_b)$",
    )
    axs[0, 0].plot(
        xline,
        helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "x")(xline),
        label=r"$p(\mu_{\alpha}|M_{tc})$",
    )
    axs[0, 0].set_ylabel("Probability")
    #  mu_dec plot
    axs[1, 1].plot(
        helperfunctions.gaussian1D(candidate_object.g2d_conv, "y")(xline),
        xline,
        label=r"$p(\mu_{\delta}|M_b)$",
    )
    axs[1, 1].plot(
        helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "y")(xline),
        xline,
        label=r"$p(\mu_{\delta}|M_{tc})$",
    )
    axs[1, 1].set_xlabel("Probability")

    max_yaxis = max(
        [
            *helperfunctions.gaussian1D(candidate_object.g2d_conv, "x")(xline),
            *helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "x")(xline),
        ]
    )
    max_xaxis = max(
        [
            *helperfunctions.gaussian1D(candidate_object.g2d_conv, "y")(xline),
            *helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "y")(xline),
        ]
    )
    textyaxis = max_yaxis
    textyaxis_step = textyaxis / 8
    #  Add lines to indicate candidate_objects position
    axs[0, 0].vlines(
        candidate_object.g2d_cc.x_mean,
        0,
        textyaxis,
        color="red",
        linestyles="dashed",
        label=r"$\mu_{true}$",
    )
    axs[1, 1].hlines(
        candidate_object.g2d_cc.y_mean,
        0,
        max_xaxis,
        color="red",
        linestyles="dashed",
        label=r"$\mu_{true}$",
    )
    #  Add text
    axs[0, 1].text(
        0,
        1 - 6 / 6,
        r"$\log_{10}\frac{{p(\mu_{{\alpha, \delta}}|M_{tc})}}{{p(\mu_{{\alpha,\delta}}|M_b)}}$"
        + f" = {candidate_object.r_tcb_pmmodel:.2f}",
    )
    axs[0, 1].text(
        0,
        1 - 5 / 6,
        r"$p(\mu_{{\alpha,\delta}}|M_b)$" + " = %.2E" % Decimal(candidate_object.p_b),
    )
    axs[0, 1].text(
        0,
        1 - 4 / 6,
        r"$p(\mu_{{\alpha,\delta}}|M_{tc})$"
        + " = %.2E" % Decimal(candidate_object.p_tc),
    )
    axs[0, 1].text(
        0,
        1 - 3 / 6,
        r"$\mu_{\alpha, \delta}$"
        + f" = [{candidate_object.g2d_cc.x_mean.value:.2f}, {candidate_object.g2d_cc.y_mean.value:.2f}]",
    )
    axs[0, 1].text(0, 1 - 2 / 6, "candidates proper motion [mas/yr]:")
    axs[0, 1].text(
        0,
        1 - 1 / 6,
        rf"$\sigma^{{M_b}}_{{pmra}}$={np.sqrt(candidate_object.cov_conv[0,0]):.0f}, $\sigma^{{M_b}}_{{pmdec}}$={np.sqrt(candidate_object.cov_conv[1,1]):.0f}",
    )

    #  Set labels and legend
    axs[1, 0].set_xlabel(r"$\mu_{\alpha} $ mas/yr")
    axs[1, 0].set_ylabel(r"$\mu_{\delta} $ mas/yr")
    fig.suptitle(
        f"Candidate is a {candidate_object.back_true} \n {target} candidate at {candidate_object.cc_true_data[band]:.2f} $mag$, final_uuid={candidate_object.final_uuid}"
    )
    # axs[0,1].remove()
    axs[0, 1].set_axis_off()
    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    #  Set all the same axis limits
    # p_lim = 0.01
    mg = np.max(
        [
            candidate_object.g2d_conv.x_mean.value
            + 5 * np.sqrt(candidate_object.cov_conv[0, 0]),
            5 * np.sqrt(candidate_object.cov_pmuM1[0, 0]),
        ]
    )
    axs[1, 0].set_xlim(-mg, mg)
    axs[0, 0].set_xlim(-mg, mg)
    mg = np.max(
        [
            candidate_object.g2d_conv.y_mean.value
            + 5 * np.sqrt(candidate_object.cov_conv[1, 1]),
            5 * np.sqrt(candidate_object.cov_pmuM1[1, 1]),
        ]
    )
    axs[1, 1].set_ylim(-mg, mg)
    plt.tight_layout()


def odds_ratio_sep_mag_plot(candidates_table, target_name, p_ratio_name):
    fig, axs = plt.subplots(1, figsize=(7, 3))
    g = sns.scatterplot(
        x="sep",
        y=p_ratio_name,
        data=candidates_table,
        hue="band",
        style="r_tcb_catalogue",
        s=60,
        ax=axs,
    )
    g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs.hlines(
        0,
        candidates_table.sep.min(),
        candidates_table.sep.max(),
        linestyles="dotted",
        colors="gray",
    )
    axs.set_ylim(
        np.percentile(candidates_table[p_ratio_name], q=20),
        candidates_table[p_ratio_name].max() + 1,
    )

    axs.set_ylabel("Odds Ratio: " + r"$\log_{10}\frac{{P(\mu|M_{tc})}}{{P(\mu|M_b)}}$")
    axs.set_xlabel(r"Separation $[mas]$")
    axs.set_title(f"Odds ratios of all candidates of {target_name}")
    plt.tight_layout()


def propagation_plot(candidate_object, host_star, axs):
    days_since_gaia = candidate_object.cc_true_data["t_days_since_Gaia"]
    years = (
        2000
        + (
            (candidate_object.cc_true_data["t_days_since_Gaia"])
            + (host_star.ref_epoch - 2000) * 365.25
        )
        / 365.25
    )
    time_days = np.arange(
        candidate_object.cc_true_data["t_days_since_Gaia"][0],
        candidate_object.cc_true_data["t_days_since_Gaia"][-1] + 1,
    )
    # Initial position
    axs.plot(
        host_star.parallax * candidate_object.plx_proj_ra[0],
        host_star.parallax * candidate_object.plx_proj_dec[0],
        "s",
        color="C0",
        label=f"{years[0]:.1f}",
    )
    for i in range(int(len(candidate_object.mean_true_companion) / 2 - 1)):
        # TRUE CANDIDATE
        axs.plot(
            candidate_object.mean_true_companion[2 + 2 * i],
            candidate_object.mean_true_companion[3 + 2 * i],
            "o",
            color="C1",
            label=rf"$M_{{tc}}$: {years[i+1]:.1f}",
        )
        helperfunctions.ellipse(
            candidate_object.mean_true_companion[2 + 2 * i],
            candidate_object.mean_true_companion[3 + 2 * i],
            candidate_object.cov_true_companion[
                2 + 2 * i : 4 + 2 * i, 2 + 2 * i : 4 + 2 * i
            ],
            "C1",
            "-",
            axs,
        )
        # BACKGROUND OBJECT
        axs.plot(
            candidate_object.mean_background_object[2 + 2 * i],
            candidate_object.mean_background_object[3 + 2 * i],
            "^",
            color="C2",
            label=rf"$M_{{b}}$: {years[i+1]:.1f}",
        )
        helperfunctions.ellipse(
            candidate_object.mean_background_object[2 + 2 * i],
            candidate_object.mean_background_object[3 + 2 * i],
            candidate_object.cov_background_object[
                2 + 2 * i : 4 + 2 * i, 2 + 2 * i : 4 + 2 * i
            ],
            "C2",
            "-",
            axs,
        )
        # MEASURED POSITION
        axs.plot(
            candidate_object.mean_measured_positions[2 + 2 * i],
            candidate_object.mean_measured_positions[3 + 2 * i],
            "s",
            color="C3",
            label=f"c: {years[i+1]:.1f}",
        )
        helperfunctions.ellipse(
            candidate_object.mean_measured_positions[2 + 2 * i],
            candidate_object.mean_measured_positions[3 + 2 * i],
            candidate_object.cov_measured_positions[
                2 + 2 * i : 4 + 2 * i, 2 + 2 * i : 4 + 2 * i
            ],
            "C3",
            "-",
            axs,
        )
    # Parallax
    time_days = np.arange(
        days_since_gaia[0],
        days_since_gaia[-1] + 1,
    )
    time_indices = [
        time_days.tolist().index(time_index) for time_index in days_since_gaia
    ]
    track_time = time_days - candidate_object.cc_true_data["t_days_since_Gaia"][0]
    x_track = helperfunctions.calc_prime_1(
        0,
        host_star.pmra,
        host_star.parallax,
        track_time / 365.25,
        candidate_object.plx_proj_ra[time_indices[0] :],
    )
    y_track = helperfunctions.calc_prime_1(
        0,
        host_star.pmdec,
        host_star.parallax,
        track_time / 365.25,
        candidate_object.plx_proj_dec[time_indices[0] :],
    )
    axs.plot(
        x_track,
        y_track,
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
    )

    axs.invert_xaxis()
    axs.legend()
    axs.set_title(
        "Relative motion of a candidate"
        + "\n"
        + rf"with $r_{{tcb}}$={candidate_object.r_tcb_2Dnmodel:.2f}"
    )
    axs.set_xlabel("RA [mas]")
    axs.set_ylabel("DEC [mas]")
    # plt.savefig(f'./plots/{target_name}_same_cov_calc.png', dpi=300, format='png')


def plot_pm_plx_binning_parameters(df_binning_parameters, catalogue_name, host_star):
    mag_min = df_binning_parameters.band.min()
    mag_max = df_binning_parameters.band.max()
    xline = np.linspace(mag_min, mag_max)
    fig, axs = plt.subplots(3, 3, figsize=(10, 7))
    for ax, col in zip(
        axs.flat,
        df_binning_parameters.drop(
            columns=[
                "band",
                "cov_pmdec_parallax",
                "cov_pmra_parallax",
                "cov_pmra_pmdec",
            ]
        ).columns,
    ):
        ax.plot(
            df_binning_parameters["band"], df_binning_parameters[col], "o", alpha=0.6
        )
        if "rho" in col:
            attribute_name = f"{col[4:]}_model_{catalogue_name}"
            popt = host_star.__getattribute__(attribute_name)
            ax.hlines(popt, mag_min, mag_max, color="C1")
        else:
            attribute_name = f"{col}_model_coeff_{catalogue_name}"
            popt = host_star.__getattribute__(attribute_name)
            attribute_name = f"{col}_model_cov_{catalogue_name}"
            pcov = host_star.__getattribute__(attribute_name)
            if len(popt) == 3:
                ax.plot(xline, popt[0] * np.exp(-xline * popt[1]) + popt[2])
                ax.fill_between(
                    xline,
                    (popt[0] + pcov[0, 0]) * np.exp(-xline * (popt[1] + pcov[1, 1]))
                    + popt[2]
                    + pcov[2, 2],
                    (popt[0] - pcov[0, 0]) * np.exp(-xline * (popt[1] - pcov[1, 1]))
                    + popt[2]
                    - pcov[2, 2],
                    alpha=0.4,
                    color="C1",
                )
            elif len(popt) == 2:
                ax.plot(xline, popt[0] * xline + popt[1], color="C1")
                ax.fill_between(
                    xline,
                    (popt[0] + pcov[0, 0]) * xline + popt[1] + pcov[1, 1],
                    (popt[0] - pcov[0, 0]) * xline + popt[1] - pcov[1, 1],
                    alpha=0.4,
                    color="C1",
                )
            elif len(popt) == 1:
                ax.hlines(popt, mag_min, mag_max, color="C1")
                ax.fill_between(xline, popt + pcov[0], popt - pcov[0], color="C1")
        ax.set_xlabel("[mag]")
        ax.set_ylabel(col)
        ax.margins(0.1, 1)
    fig.suptitle("Model fitting parameters")
    plt.tight_layout()
