#  Making plots
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.markers as mmarkers

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
            candidate_object.g2d_conv, major_axis, minor_axis, angle, "black", axs[1, 0]
        )

    for confd in [0.5, 0.9, 0.99]:
        major_axis, minor_axis, angle = helperfunctions.get_ellipse_props(
            candidate_object.cov_pmuM1, confidence=confd
        )
        helperfunctions.add_ellp_patch(
            candidate_object.g2d_pmuM1, major_axis, minor_axis, angle, "#56B4E9", axs[1, 0]
        )

    axs[1, 0].plot(
        candidate_object.g2d_cc.x_mean,
        candidate_object.g2d_cc.y_mean,
        color="#E69F00",
        marker="x",
        label="candidate",
    )
    #  mu_ra plot
    axs[0, 0].plot(
        xline,
        helperfunctions.gaussian1D(candidate_object.g2d_conv, "x")(xline),
        label=r"$p(\mu_{\alpha}|M_b)$",
        color="black"
    )
    axs[0, 0].plot(
        xline,
        helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "x")(xline),
        label=r"$p(\mu_{\alpha}|M_{tc})$",
        color="#56B4E9",
    )
    axs[0, 0].set_ylabel("Probability")
    #  mu_dec plot
    axs[1, 1].plot(
        helperfunctions.gaussian1D(candidate_object.g2d_conv, "y")(xline),
        xline,
        label=r"$p(\mu_{\delta}|M_b)$",
        color="black"
    )
    axs[1, 1].plot(
        helperfunctions.gaussian1D(candidate_object.g2d_pmuM1, "y")(xline),
        xline,
        label=r"$p(\mu_{\delta}|M_{tc})$",
        color="#56B4E9",
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
        color="#E69F00",
        linestyles="dashed",
        label=r"$\mu_{true}$",
    )
    axs[1, 1].hlines(
        candidate_object.g2d_cc.y_mean,
        0,
        max_xaxis,
        color="#E69F00",
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
    axs00_ylim = axs[1,0].get_ylim()
    axs[1, 1].set_ylim(axs00_ylim)
    plt.tight_layout()


def odds_ratio_sep_mag_plot(candidates_table, target_name, p_ratio_name):
    """Creates a odds ratio vs. separation plot by magnitude and catalogue
    Args:
        candidates_table (pandas.DataFrame): Table with the candidates data.
        target_name (str): Name of the host star.
        p_ratio_name (str): r_tcb_pmmodel or r_tcb_2Dnmodel.
    """
    candidates_table = candidates_table.rename(columns={"band": "K-band"})
    candidates_table["K-band"] = candidates_table["K-band"].round(0)
    fig, axs = plt.subplots(1, figsize=(7, 3))
    g = sns.scatterplot(
        x="sep",
        y=p_ratio_name,
        data=candidates_table,
        hue="K-band",
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

    axs.set_ylabel("Odds Ratio: " + r"$\log_{10}\frac{{P(\mu|M_{tc})}}{{P(\mu|M_b)}}$")
    axs.set_xlabel(r"Separation $[mas]$")
    axs.set_title(f"Odds ratios of all candidates of {target_name}")
    axs.margins(0.15, 0.2)
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


def p_ratio_relative_position(
    candidate_df,
    target_name,
    markers=list(mmarkers.MarkerStyle.markers.keys())[2:],
    colors=[
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ],
):
    epochs = candidate_df["ref_epochs"].to_list()[0]
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(5, 5),
        gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 3]},
    )
    # Ellipses
    # need to be plotet by epochs
    # TRUE COMPANION
    # First epoch equals all following epochs of the true companion
    # Only one epoch is displayed
    means = candidate_df["mean_true_companion"].to_list()[0]  # [2:]
    cov = np.array(candidate_df["cov_true_companion"].to_list()[0])
    dras = means[::2]
    ddecs = means[1::2]
    axs[1, 0].plot(
        dras[0],
        ddecs[0],
        markers[0],
        color=colors[2],
        alpha=1,
    )
    helperfunctions.ellipse(
        dras[0],
        ddecs[0],
        cov[:2, :2],
        color=colors[2],
        linestyle="-",
        linewidth=1,
        alpha=0.6,
        axis=axs[1, 0],
        zorder=0,
    )

    for no_obs in range(1, len(epochs)):
        for i, colordirection in enumerate(
            zip(colors, ["background_object", "measured_positions"])
        ):
            color, direction = colordirection
            means = candidate_df[f"mean_{direction}"].to_list()[0]  # [2:]
            cov = np.array(candidate_df[f"cov_{direction}"].to_list()[0])
            dras = means[::2]
            ddecs = means[1::2]
            axs[1, 0].plot(
                dras[no_obs],
                ddecs[no_obs],
                markers[no_obs],
                color=color,
                alpha=1,
            )
            helperfunctions.ellipse(
                dras[no_obs],
                ddecs[no_obs],
                cov[2 * no_obs : 2 + 2 * no_obs, 2 * no_obs : 2 + 2 * no_obs],
                color=color,
                linestyle="-",
                linewidth=1,
                alpha=0.6,
                axis=axs[1, 0],
                zorder=0,
            )
    for no_obs in range(1, len(epochs)):
        for i, colordirection in enumerate(
            zip(colors, ["background_object", "measured_positions", "true_companion"])
        ):
            color, direction = colordirection
            means = candidate_df[f"mean_{direction}"].to_list()[0]  # [2:]
            cov = np.array(candidate_df[f"cov_{direction}"].to_list()[0])
            dras = means[::2]
            ddecs = means[1::2]
            g2d_bg = helperfunctions.Gaussian2D(
                x_mean=dras[no_obs],
                y_mean=ddecs[no_obs],
                cov_matrix=cov[
                    2 * no_obs : 2 + 2 * no_obs, 2 * no_obs : 2 + 2 * no_obs
                ],
            )
            g1d = helperfunctions.gaussian1D(g2d_bg, "x")
            xline = np.linspace(axs[1, 0].get_xlim()[0], axs[1, 0].get_xlim()[1], 200)
            axs[0, 0].plot(
                xline,
                g1d(xline),
                color=color,
                alpha=0.8,
            )
            g1d = helperfunctions.gaussian1D(g2d_bg, "y")
            xline = np.linspace(axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], 200)
            axs[1, 1].plot(
                g1d(xline),
                xline,
                color=color,
                alpha=0.8,
            )
    axs[1, 0].set_xlabel(r"$\Delta RA$ [mas]")
    axs[1, 0].set_ylabel(r"$\Delta DEC$ [mas]")
    axs[0, 0].axis("off")
    axs[1, 1].axis("off")
    axs[0, 1].axis("off")

    axs[1, 0].spines["top"].set_visible(False)
    axs[1, 0].spines["right"].set_visible(False)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    legend_elements = [
        plt.Line2D([0], [0], marker=marker, color="black", label=label, linestyle="")
        for marker, label in zip(markers, epochs)
    ]
    axs[1, 0].legend(
        handles=legend_elements, loc="lower right", title="Epochs", fontsize=8
    )
    legend_elements = [
        plt.Line2D([0], [0], color=color, label=label, linestyle="-")
        for color, label in zip(colors, ["Background", "Measured", "Companion"])
    ]
    axs[0, 1].legend(handles=legend_elements, loc="right", title="Type", fontsize=8)
    fig.suptitle(
        f"Relative Position: Candidate ({candidate_df['final_uuid'].unique()[0]}) - {target_name} \n  $\log_{{10}} r_{{tcb}}={candidate_df['r_tcb_2Dnmodel'].unique()[0]:.2f}$"
    )
    return fig, axs
