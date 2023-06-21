import itertools
import logging
import re

import tqdm
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from requests import exceptions
from scipy import optimize

from compass import helperfunctions


class CovarianceMatrix:
    def cov_pmra_pmdec(model_object):
        return (
            model_object.pmra_error
            * model_object.pmdec_error
            * model_object.pmra_pmdec_corr
        )

    def cov_pmdirection_plx(model_object, pmdirection):
        return (
            model_object.__getattribute__(pmdirection)
            * model_object.parallax_error
            * model_object.parallax_pmra_corr
        )

    def calc_variance_x(time, plx_proj, host_star, backgroundmodel):
        cov_plx_pm_h = (
            host_star.pmra_error
            * host_star.parallax_error
            * host_star.parallax_pmra_corr
        )
        cov_plx_pm_b = (
            backgroundmodel.pmra_error
            * backgroundmodel.parallax_error
            * backgroundmodel.parallax_pmra_corr
        )
        var_prime = (
            time**2 * (backgroundmodel.pmra_error**2 + host_star.pmra_error**2)
            + plx_proj**2
            * (backgroundmodel.parallax_error**2 + host_star.parallax_error**2)
            + 2 * time * plx_proj * (cov_plx_pm_b + cov_plx_pm_h)
        )
        return var_prime

    def calc_variance_y(time, plx_proj, host_star, backgroundmodel):
        cov_plx_pm_h = (
            host_star.pmdec_error
            * host_star.parallax_error
            * host_star.parallax_pmra_corr
        )
        cov_plx_pm_b = (
            backgroundmodel.pmdec_error
            * backgroundmodel.parallax_error
            * backgroundmodel.parallax_pmra_corr
        )
        var_prime = (
            time**2 * (backgroundmodel.pmdec_error**2 + host_star.pmdec_error**2)
            + plx_proj**2
            * (backgroundmodel.parallax_error**2 + host_star.parallax_error**2)
            + 2 * time * plx_proj * (cov_plx_pm_b + cov_plx_pm_h)
        )
        return var_prime

    def calc_covariance_xiyi(time, plx_proj_x, plx_proj_y, host_star, backgroundmodel):
        cov_pmra_b_pmdec_b = (
            backgroundmodel.pmra_error
            * backgroundmodel.pmdec_error
            * backgroundmodel.pmra_pmdec_corr
        )
        cov_pmra_h_pmdec_h = (
            host_star.pmra_error * host_star.pmdec_error * host_star.pmra_pmdec_corr
        )
        cov_plx_b_pmra_b = (
            backgroundmodel.parallax_error
            * backgroundmodel.pmra_error
            * backgroundmodel.parallax_pmra_corr
        )
        cov_plx_h_pmra_h = (
            host_star.parallax_error
            * host_star.pmra_error
            * host_star.parallax_pmra_corr
        )
        cov_plx_b_pmdec_b = (
            backgroundmodel.parallax_error
            * backgroundmodel.pmdec_error
            * backgroundmodel.parallax_pmdec_corr
        )
        cov_plx_h_pmdec_h = (
            host_star.parallax_error
            * host_star.pmdec_error
            * host_star.parallax_pmdec_corr
        )
        cov = (
            time**2 * (cov_pmra_b_pmdec_b + cov_pmra_h_pmdec_h)
            + plx_proj_x
            * plx_proj_y
            * (backgroundmodel.parallax_error**2 + host_star.parallax_error**2)
            + time * plx_proj_y * (cov_plx_b_pmra_b + cov_plx_h_pmra_h)
            + time * plx_proj_x * (cov_plx_b_pmdec_b + cov_plx_h_pmdec_h)
        )
        return cov

    def calc_covariance_xixj(
        timei, timej, plx_proj_ra_i, plx_proj_ra_j, host_star, backgroundmodel
    ):
        cov_plx_b_pmra_b = (
            backgroundmodel.parallax_error
            * backgroundmodel.pmra_error
            * backgroundmodel.parallax_pmra_corr
        )
        cov_plx_h_pmra_h = (
            host_star.parallax_error
            * host_star.pmra_error
            * host_star.parallax_pmra_corr
        )
        cov = (
            timei
            * timej
            * (backgroundmodel.pmra_error**2 + host_star.pmra_error**2)
            + timej * plx_proj_ra_i * (cov_plx_b_pmra_b + cov_plx_h_pmra_h)
            + timei * plx_proj_ra_j * (cov_plx_b_pmra_b + cov_plx_h_pmra_h)
            + plx_proj_ra_j
            * plx_proj_ra_i
            * (backgroundmodel.parallax_error**2 + host_star.parallax_error**2)
        )
        return cov

    def calc_covariance_yiyj(
        timei, timej, plx_proj_dec_i, plx_proj_dec_j, host_star, backgroundmodel
    ):
        cov_plx_b_pmdec_b = (
            backgroundmodel.parallax_error
            * backgroundmodel.pmdec_error
            * backgroundmodel.parallax_pmdec_corr
        )
        cov_plx_h_pmdec_h = (
            host_star.parallax_error
            * host_star.pmdec_error
            * host_star.parallax_pmdec_corr
        )
        cov = (
            timei
            * timej
            * (backgroundmodel.pmdec_error**2 + host_star.pmdec_error**2)
            + timej * plx_proj_dec_i * (cov_plx_b_pmdec_b + cov_plx_h_pmdec_h)
            + timei * plx_proj_dec_j * (cov_plx_b_pmdec_b + cov_plx_h_pmdec_h)
            + plx_proj_dec_j
            * plx_proj_dec_i
            * (backgroundmodel.parallax_error**2 + host_star.parallax_error**2)
        )
        return cov

    def calc_covariance_xiyj(
        timei, timej, plx_proj_ra_i, plx_proj_dec_j, host_star, backgroundmodel
    ):
        cov_pm = CovarianceMatrix.cov_pmra_pmdec(
            backgroundmodel
        ) + CovarianceMatrix.cov_pmra_pmdec(host_star)
        cov_pmra_plx = CovarianceMatrix.cov_pmdirection_plx(
            backgroundmodel, "pmra"
        ) + CovarianceMatrix.cov_pmdirection_plx(host_star, "pmra")
        cov_plx_pmdec = CovarianceMatrix.cov_pmdirection_plx(
            backgroundmodel, "pmdec"
        ) + CovarianceMatrix.cov_pmdirection_plx(host_star, "pmdec")
        cov_plx = backgroundmodel.parallax_error**2 + host_star.parallax_error**2
        cov = (
            timei * timej * cov_pm
            + timei * plx_proj_dec_j * cov_pmra_plx
            + timej * plx_proj_ra_i * cov_plx_pmdec
            + plx_proj_ra_i * plx_proj_dec_j * cov_plx
        )

        return cov

    def calc_covariance_yixj(
        timei, timej, plx_proj_dec_i, plx_proj_ra_j, host_star, backgroundmodel
    ):
        cov_pm = CovarianceMatrix.cov_pmra_pmdec(
            backgroundmodel
        ) + CovarianceMatrix.cov_pmra_pmdec(host_star)
        cov_pmra_plx = CovarianceMatrix.cov_pmdirection_plx(
            backgroundmodel, "pmra"
        ) + CovarianceMatrix.cov_pmdirection_plx(host_star, "pmra")
        cov_plx_pmdec = CovarianceMatrix.cov_pmdirection_plx(
            backgroundmodel, "pmdec"
        ) + CovarianceMatrix.cov_pmdirection_plx(host_star, "pmdec")
        cov_plx = backgroundmodel.parallax_error**2 + host_star.parallax_error**2
        cov = (
            timej * timei * cov_pm
            + timej * plx_proj_dec_i * cov_pmra_plx
            + timei * plx_proj_ra_j * cov_plx_pmdec
            + plx_proj_ra_j * plx_proj_dec_i * cov_plx
        )
        return cov

    def cov_propagation(C_0, days_1, days_2, plx_proj_x, plx_proj_y, host_star):
        time = (days_2 - days_1) / 365.35
        var_x_prime_2 = CovarianceMatrix.calc_variance_x(time, plx_proj_x, host_star)
        var_y_prime_2 = CovarianceMatrix.calc_variance_y(time, plx_proj_y, host_star)
        cov_x_prime_2_y_prime_2 = CovarianceMatrix.calc_covariance_xiyi(
            time, plx_proj_x, plx_proj_y, host_star
        )
        C_prime_2 = np.array(
            [
                [var_x_prime_2, cov_x_prime_2_y_prime_2],
                [cov_x_prime_2_y_prime_2, var_y_prime_2],
            ]
        )
        C = C_0 + C_prime_2
        return C

    def covariance_matrix(
        times,
        plx_proj_ra,
        plx_proj_dec,
        host_star,
        backgroundmodel,
        return_dict=False,
    ):
        dict_var_covar = {}
        time_days = np.arange(
            times[0],
            times[-1] + 1,
        )
        time_indices = [time_days.tolist().index(time_index) for time_index in times]
        J = list(range(1, len(times) + 1))
        for i, time in enumerate(times):
            dict_var_covar[f"var_x{i+1}"] = CovarianceMatrix.calc_variance_x(
                time / 365.25, plx_proj_ra[time_indices[i]], host_star, backgroundmodel
            )
            dict_var_covar[f"var_y{i+1}"] = CovarianceMatrix.calc_variance_y(
                time / 365.25, plx_proj_dec[time_indices[i]], host_star, backgroundmodel
            )
            for j in J:
                if i + 1 != j:
                    dict_var_covar[
                        f"covar_x{i+1}x{j}"
                    ] = CovarianceMatrix.calc_covariance_xixj(
                        times[i] / 365.25,
                        times[j - 1] / 365.25,
                        plx_proj_ra[time_indices[i]],
                        plx_proj_ra[time_indices[j - 1]],
                        host_star,
                        backgroundmodel,
                    )
                    dict_var_covar[
                        f"covar_y{i+1}y{j}"
                    ] = CovarianceMatrix.calc_covariance_yiyj(
                        times[i] / 365.25,
                        times[j - 1] / 365.25,
                        plx_proj_dec[time_indices[i]],
                        plx_proj_dec[time_indices[j - 1]],
                        host_star,
                        backgroundmodel,
                    )
                dict_var_covar[
                    f"covar_x{i+1}y{j}"
                ] = CovarianceMatrix.calc_covariance_xiyi(
                    times[j - 1] / 365.25,
                    plx_proj_ra[time_indices[j - 1]],
                    plx_proj_dec[time_indices[j - 1]],
                    host_star,
                    backgroundmodel,
                )
                if f"covar_x{i+1}y{j}" in dict_var_covar.keys() and i + 1 == j:
                    continue
                else:
                    dict_var_covar[
                        f"covar_x{i+1}y{j}"
                    ] = CovarianceMatrix.calc_covariance_xiyj(
                        times[i] / 365.25,
                        times[j - 1] / 365.25,
                        plx_proj_ra[time_indices[i]],
                        plx_proj_dec[time_indices[j - 1]],
                        host_star,
                        backgroundmodel,
                    )
                    dict_var_covar[
                        f"covar_y{i+1}x{j}"
                    ] = CovarianceMatrix.calc_covariance_yixj(
                        times[i] / 365.25,
                        times[j - 1] / 365.25,
                        plx_proj_dec[time_indices[i]],
                        plx_proj_ra[time_indices[j - 1]],
                        host_star,
                        backgroundmodel,
                    )
            J = J[1:]
        var_keys = [el for el in list(dict_var_covar.keys()) if el.startswith("var")]
        empty_matrix = np.zeros((len(var_keys), len(var_keys)))
        np.fill_diagonal(
            empty_matrix, np.array([dict_var_covar[var_key] for var_key in var_keys])
        )
        sorted_data = dict(
            sorted(
                dict_var_covar.items(),
                key=lambda x: (int(x[0][-1]), x[0][-4], x[0][-2]),
            )
        )
        covar_keys = sorted(
            list(
                set(
                    [
                        el[:8]
                        for el in list(dict_var_covar.keys())
                        if el.startswith("covar")
                    ]
                )
            ),
            key=lambda x: (x[-1], x[-2]),
        )
        for i, key in enumerate(covar_keys):
            empty_matrix[i, i + 1 :] = list(
                {k: v for k, v in sorted_data.items() if k.startswith(key)}.values()
            )
        sym_matrix = empty_matrix + empty_matrix.T - np.diag(empty_matrix.diagonal())
        if return_dict:
            return dict_var_covar, sym_matrix
        return sym_matrix


class Candidate:
    """Model, true data and likelihoods, p_ratios of one candidate.

    Attributes:
        cc_true_data (dict): True data from df_survey.
        cc_true_data (dict): Model data based on host star fits.
        g2d_model (astropy.modeling.functional_models.Gaussian2D): 2D Gaussian of the model.
        g2d_conv (astropy.modeling.functional_models.Gaussian2D): 2D Gaussian of the convolution.
        g2d_cc (astropy.modeling.functional_models.Gaussian2D): 2D Gaussian of the candidate.
        g2d_pmuM1 (astropy.modeling.functional_models.Gaussian2D): 2D Gaussian of the candidate at (0,0).
        cov_model (numpy.array): 2x2 covariance matrix.
        cov_cc (numpy.array): 2x2 covariance matrix.
        cov_conv (numpy.array): 2x2 covariance matrix.
        cov_pmuM1 (numpy.array): 2x2 covariance matrix.
        p_b (float): Odd for being a background object.
        p_ratio (float): Odds ratio.
        p_tc (float): Odd for being a true companion.
        back_true (str): true companion or background object.
        mean_measured_positions (numpy.darray): Measured position of candidate.
        mean_true_companion (numpy.darray): Calculated position of candidate by pm and plx of star.
        mean_background_object (numpy.darray): Calculated position of candidate by pm and plx of backgorund model.
        cov_measured_positions (numpy.darray): Covariance matrix of measured position of candidate
        cov_true_companion (numpy.darray): Covariance matrix of candidate by pm and plx of star.
        cov_background_object (numpy.darray): Covariance matrix of candidate by pm and plx of backgorund model.
        r_tcb_2Dnmodel (float): log10(P_tc / P_b).
        r_tcb_pmmodel (float): log10(P_tc / P_b).
    """

    def __init__(self, df_survey, index_candidate, host_star, band, catalogue):
        """Init candidates.

        Args:
            df_survey (pandas.DataFrame): Data of the candidates of a single host star.
            index_candidate (int): index integer of the candidate in df_survey.
            host_star (Class Object): Previously initiated class for the host star.
            band (str): Band which the candidate was observed in df_survey (columnname).
            catalogue (str): Name of the catalogue the model is based on: gaia, gaiacalctmass or tmass.
        """
        cc_true_data = df_survey.iloc[index_candidate][
            [
                "dRA",
                "dDEC",
                "dRA_err",
                "dDEC_err",
                "dRA_dDEC_corr",
                "t_days_since_Gaia",
                "pmra_mean",
                "pmdec_mean",
                "pmra_abs",
                "pmdec_abs",
                "pmra_pmdec_corr",
                "pmra_error",
                "pmdec_error",
                "final_uuid",
                "sep",
                band,
            ]
        ].to_dict()

        pm_options = ["pmra", "pmdec"]
        y_options = ["mean", "stddev"]
        cc_pm_background_data = {}
        for pm_value in pm_options:
            for y_option in y_options:
                column = f"{pm_value}_{y_option}_model_coeff_{catalogue}"
                background_model_parameters = host_star.__getattribute__(column)
                if y_option == "mean" or len(background_model_parameters) == 2:
                    cc_pm_background_data[pm_value + "_" + y_option] = (
                        background_model_parameters[0] * cc_true_data[band]
                        + background_model_parameters[1]
                    )
                elif y_option == "stddev":
                    if len(background_model_parameters) == 3:
                        cc_pm_background_data[pm_value + "_" + y_option] = (
                            background_model_parameters[0]
                            * np.exp(
                                -background_model_parameters[1] * cc_true_data[band]
                            )
                            + background_model_parameters[2]
                        )
                    else:
                        cc_pm_background_data[pm_value + "_" + y_option] = (
                            background_model_parameters[0] * cc_true_data[band]
                            + background_model_parameters[1]
                        )
        column = f"pmra_pmdec_model_{catalogue}"
        cc_pm_background_data["rho_mean"] = host_star.__getattribute__(column)
        cc_pm_background_data["rho_stddev"] = host_star.__getattribute__(column)
        self.cc_true_data = cc_true_data
        self.cc_pm_background_data = cc_pm_background_data
        self.final_uuid = cc_true_data["final_uuid"]

        # Parallax projections coordinates
        time_days_from_equinox = np.arange(
            int(self.cc_true_data["t_days_since_Gaia"][0]),
            int(self.cc_true_data["t_days_since_Gaia"][-1] + 1),
        )
        plx_proj_ra, plx_proj_dec = helperfunctions.parallax_projection(
            time_days_from_equinox / 365.25, host_star
        )
        self.plx_proj_ra = plx_proj_ra
        self.plx_proj_dec = plx_proj_dec

    def calc_likelihoods_2Dnmodel(self, host_star, catalogue_name="gaiacalctmass"):
        """Attributes the likelihoods to the candidate object in terms of the means and covariance matrices.

        Args:
            host_star (class object): Use of proper motion and parallax of the star.
            background (class object): Use of proper motion and parallax of the backgorund distribution.
        """
        cc_true_data = self.cc_true_data
        days_since_gaia = cc_true_data["t_days_since_Gaia"]

        # Initiate Background Class Object
        candidate_mag = cc_true_data["band"]
        background_model = BackgroundModel(
            candidate_mag, host_star, catalogue_name=catalogue_name
        )

        # Calculate the means of the distributions
        time_obs_days_from_gaia = np.array(self.cc_true_data["t_days_since_Gaia"])
        time_days = np.arange(
            self.cc_true_data["t_days_since_Gaia"][0],
            self.cc_true_data["t_days_since_Gaia"][-1] + 1,
        )
        candidate_position = [cc_true_data["dRA"], cc_true_data["dDEC"]]
        mean_obs, mean_tc, mean_b = (
            [candidate_position[0][0], candidate_position[1][0]] for i in range(3)
        )
        for i, time_gaia in enumerate(time_obs_days_from_gaia[1:]):
            delta_time = time_gaia - time_obs_days_from_gaia[0]
            time_gaia_index = time_days.tolist().index(time_gaia)
            x_tc = candidate_position[0][0]
            y_tc = candidate_position[1][0]
            mean_tc.append(x_tc)
            mean_tc.append(y_tc)
            x_b = (
                x_tc
                + (background_model.pmra - host_star.pmra) * delta_time / 365.25
                + (background_model.parallax - host_star.parallax)
                * self.plx_proj_ra[time_gaia_index]
            )
            y_b = (
                y_tc
                + (background_model.pmdec - host_star.pmdec) * delta_time / 365.25
                + (background_model.parallax - host_star.parallax)
                * self.plx_proj_dec[time_gaia_index]
            )
            mean_b.append(x_b)
            mean_b.append(y_b)
            # The second measured posiiton is the difference of both measurements
            # Because the first position is out zero point and the candidate deviates from
            # that position at the second observation by the difference.
            # If the difference would be zero, the candidate would have reside at the same position
            # relative to the host star.
            x_c = candidate_position[0][i + 1]
            y_c = candidate_position[1][i + 1]
            mean_obs.append(x_c)
            mean_obs.append(y_c)
        mean_obs = np.array([float(i) for i in mean_obs])
        mean_tc = np.array([float(i) for i in mean_tc])
        mean_b = np.array([float(i) for i in mean_b])
        self.mean_measured_positions = mean_obs
        self.mean_true_companion = mean_tc
        self.mean_background_object = mean_b

        # Covariance matrix for the measured position
        cov_obs = {}
        for row in range(len(cc_true_data["dRA_err"])):
            cov_obs[f"cov_{row}"] = np.array(
                [
                    [
                        cc_true_data["dRA_err"][row] ** 2,
                        cc_true_data["dRA_dDEC_corr"][row]
                        * cc_true_data["dRA_err"][row]
                        * cc_true_data["dDEC_err"][row],
                    ],
                    [
                        cc_true_data["dRA_dDEC_corr"][row]
                        * cc_true_data["dRA_err"][row]
                        * cc_true_data["dDEC_err"][row],
                        cc_true_data["dDEC_err"][row] ** 2,
                    ],
                ]
            )
        empty_matrix = np.zeros((int(len(mean_obs)), int(len(mean_obs))))
        i = 0
        for key in cov_obs.keys():
            empty_matrix[i : 2 + i][:, i : 2 + i] = cov_obs[key]
            i += 2
        self.cov_measured_positions = empty_matrix

        # Covariance matrix for the true companion model
        cov_obs = {}
        cov_obs = np.array(
            [
                [
                    cc_true_data["dRA_err"][0] ** 2,
                    cc_true_data["dRA_dDEC_corr"][0]
                    * cc_true_data["dRA_err"][0]
                    * cc_true_data["dDEC_err"][0],
                ],
                [
                    cc_true_data["dRA_dDEC_corr"][0]
                    * cc_true_data["dRA_err"][0]
                    * cc_true_data["dDEC_err"][0],
                    cc_true_data["dDEC_err"][0] ** 2,
                ],
            ]
        )
        empty_matrix = np.zeros((int(len(mean_obs)), int(len(mean_obs))))
        i = 0
        for key in range(int(empty_matrix.shape[0] / 2)):
            empty_matrix[i : 2 + i][:, i : 2 + i] = cov_obs
            i += 2
        self.cov_true_companion = empty_matrix

        # Covariance matrix for being background object
        sigma_prime_b = CovarianceMatrix.covariance_matrix(
            days_since_gaia,
            self.plx_proj_ra,
            self.plx_proj_dec,
            host_star,
            background_model,
        )
        self.cov_background_object = self.cov_measured_positions + sigma_prime_b

    def calc_likelihoods_pmmodel(self, host_star, sigma_model_min, sigma_cc_min):
        """Attributes the likelihoods to the candidate object in terms
        of the means and covariance matrices.

        Args:
            host_star (class): Previously initiated class for the host star.
            sigma_model_min (float): The inflating factor for the model likelihood.
            sigma_cc_min (float): The inflating factor for its likelihood.
        """
        g2d_model, cov_model = helperfunctions.get_g2d_func(
            self.cc_pm_background_data["pmra_mean"] - host_star.pmra,
            self.cc_pm_background_data["pmdec_mean"] - host_star.pmdec,
            np.sqrt(
                self.cc_pm_background_data["pmra_stddev"] ** 2 + sigma_model_min**2
            ),
            np.sqrt(
                self.cc_pm_background_data["pmdec_stddev"] ** 2 + sigma_model_min**2
            ),
            self.cc_pm_background_data["rho_mean"],
        )
        g2d_cc, cov_cc = helperfunctions.get_g2d_func(
            self.cc_true_data["pmra_mean"],
            self.cc_true_data["pmdec_mean"],
            np.sqrt(self.cc_true_data["pmra_error"] ** 2 + sigma_cc_min**2),
            np.sqrt(self.cc_true_data["pmdec_error"] ** 2 + sigma_cc_min**2),
            self.cc_true_data["pmra_pmdec_corr"],
        )
        g2d_conv, cov_conv = helperfunctions.convolution2d(
            np.array(
                [
                    [self.cc_pm_background_data["pmra_mean"] - host_star.pmra],
                    [self.cc_pm_background_data["pmdec_mean"] - host_star.pmdec],
                ]
            ),
            np.array([[0], [0]]),
            cov_model,
            cov_cc,
        )
        g2d_pmuM1, cov_pmuM1 = helperfunctions.get_g2d_func(
            0,
            0,
            np.sqrt(self.cc_true_data["pmra_error"] ** 2 + sigma_cc_min**2),
            np.sqrt(self.cc_true_data["pmdec_error"] ** 2 + sigma_cc_min**2),
            rho=0,
        )
        self.g2d_model = g2d_model
        self.g2d_conv = g2d_conv
        self.g2d_cc = g2d_cc
        self.g2d_pmuM1 = g2d_pmuM1
        self.cov_model = cov_model
        self.cov_cc = cov_cc
        self.cov_conv = cov_conv
        self.cov_pmuM1 = cov_pmuM1

    def calc_prob_ratio_2Dnmodel(self):
        """Calculates the odds ratio based on the pm and plx model."""
        P_tc = helperfunctions.n_dim_gauss_evaluated(
            self.mean_measured_positions,
            self.mean_true_companion,
            self.cov_true_companion,
        )
        P_b = helperfunctions.n_dim_gauss_evaluated(
            self.mean_measured_positions,
            self.mean_background_object,
            self.cov_background_object,
        )
        if P_b == 0 and P_tc > 0:
            r_tcb = 300
        elif P_b == 0 and P_tc == 0:
            r_tcb = 0
        elif P_tc > 0 and P_b > 0:
            r_tcb = np.log10(P_tc / P_b)
        else:
            r_tcb = 0
        self.r_tcb_2Dnmodel = r_tcb

    def calc_prob_ratio_pmmodel(self):
        """Calculates the odds ratio based on the modelled g2d functions."""
        #  Calculate ratio and statement
        #  Adding the text to top right panel
        P_b = self.g2d_conv(
            self.cc_true_data["pmra_mean"], self.cc_true_data["pmdec_mean"]
        )[0]
        P_tc = self.g2d_pmuM1(
            self.cc_true_data["pmra_mean"], self.cc_true_data["pmdec_mean"]
        )
        if P_b == 0:
            r_tcb = 1
        elif P_tc > 0 and P_b > 0:
            r_tcb = np.log10(P_tc / P_b)
        else:
            r_tcb = 0
        if r_tcb > 0:
            back_true = "true companion"
        else:
            back_true = "background object"
        self.p_b = P_b
        self.p_tc = P_tc
        self.r_tcb_pmmodel = r_tcb
        self.back_true = back_true


class HostStar:
    """Host star of the candidates. Properties of the host star have the units given in gaiadr3.gaia_source.

    Attributes:
        ra (float): Properties of the host star.
        ra_error (float): Properties of the host star.
        dec (float): Properties of the host star.
        dec_error (float): Properties of the host star.
        ref_epoch (float): Properties of the host star.
        parallax (float): Properties of the host star.
        parallax_error (float): Properties of the host star.
        pmra (float): Properties of the host star.
        pmdec (float): Properties of the host star.
        pmra_error (float): Properties of the host star.
        pmdec_error (float): Properties of the host star.
        pmra_pmdec_corr (float): Properties of the host star.
        parallax_pmra_corr (float): Properties of the host star.
        parallax_pmdec_corr (float): Properties of the host star.
        phot_g_mean_mag (float): Properties of the host star.
        phot_bp_mean_mag (float): Properties of the host star.
        phot_rp_mean_mag (float): Properties of the host star.
        object_found (Boolean): Boolean whether the object was found.
        cone_tmass_cross (pandas.DataFrame): Containing the cone cross matched objects.
        cone_tmass_cross (pandas.DataFrame): containing the cone Gaia objects.
        candidates (pandas.DataFrame): Containing id, p_ratio and p_ratio_catalogue.
    """

    logger = logging.getLogger("astroquery")
    logger.setLevel(logging.ERROR)

    def __init__(self, target):
        """
        Searches for the given target id in the Simbad database
        for the Gaia source id and returns the data on the star.

        Args:
            target (str): Name of the target.
        """
        logger = logging.getLogger("astroquery")
        customSimbad = Simbad()
        customSimbad.add_votable_fields("ids")
        simbad_object_ids = customSimbad.query_object(target)["IDS"][0].split("|")
        if len(simbad_object_ids) >= 1:
            for identifier in simbad_object_ids:
                if "Gaia DR3" in identifier:
                    source_id = identifier[8:]
                    catalogue = "gaiadr3.gaia_source"
                    break
                elif "Gaia DR2" in identifier:
                    source_id = identifier[8:]
                    catalogue = "gaiadr2.gaia_source"
                elif "Gaia DR1" in identifier:
                    source_id = identifier[8:]
                    catalogue = "gaiadr1.gaia_source"
                else:
                    catalogue = False
                    self.object_found = False
            if catalogue:
                sql_query = f"""SELECT ra, ra_error, dec, dec_error, ref_epoch,
                parallax, parallax_error, pmra, pmdec, pmra_error,
                pmdec_error, pmra_pmdec_corr, parallax_pmra_corr,
                parallax_pmdec_corr, phot_g_mean_mag,
                phot_bp_mean_mag, phot_rp_mean_mag
                    FROM {catalogue}
                    WHERE source_id={source_id}
                    """
                try:
                    # Fetch the GAIA data with source_id from target name
                    job = Gaia.launch_job_async(sql_query)
                    target_data = job.get_results().to_pandas()
                    self.ra = target_data.ra[0]
                    self.ra_error = target_data.ra_error[0]
                    self.dec = target_data.dec[0]
                    self.dec_error = target_data.dec_error[0]
                    self.ref_epoch = target_data.ref_epoch[0]
                    self.parallax = target_data.parallax[0]
                    self.parallax_error = target_data.parallax_error[0]
                    self.pmra = target_data.pmra[0]
                    self.pmdec = target_data.pmdec[0]
                    self.pmra_error = target_data.pmra_error[0]
                    self.pmdec_error = target_data.pmdec_error[0]
                    self.pmra_pmdec_corr = target_data.pmra_pmdec_corr[0]
                    self.parallax_pmra_corr = target_data.parallax_pmra_corr[0]
                    self.parallax_pmdec_corr = target_data.parallax_pmdec_corr[0]
                    self.phot_g_mean_mag = target_data.phot_g_mean_mag[0]
                    self.phot_bp_mean_mag = target_data.phot_bp_mean_mag[0]
                    self.phot_rp_mean_mag = target_data.phot_rp_mean_mag[0]
                    self.object_found = True
                    if np.isnan(target_data.pmra[0]) or np.isnan(target_data.pmdec[0]):
                        self.object_found = False
                        logger.info(f"{target}: Proper motion is missing in Gaia.")
                except exceptions.HTTPError as hperr:
                    logger.error("Received exceptions.HTTPError", hperr)
        else:
            self.object_found = False
            logger.error("Not found in Simbad.")

    def cone_tmasscross_objects(self, cone_radius):
        """Cone around the target star and cross match with 2MASS.

        Args:
            cone_radius (float): Search cone radius in degree.
        """
        if self.object_found:
            job = Gaia.launch_job_async(
                f"""SELECT 
                gaia.source_id as source_id,
                gaia.ra as ra ,
                gaia.ra_error as ra_error,
                gaia.dec as dec,
                gaia.dec_error as dec_error,
                gaia.ref_epoch as ref_epoch,
                gaia.parallax as parallax,
                gaia.parallax_error as parallax_error,
                gaia.pmra as pmra,
                gaia.pmdec as pmdec,
                gaia.pmra_error as pmra_error,
                gaia.pmdec_error as pmdec_error,
                gaia.pmra_pmdec_corr as pmra_pmdec_corr,
                gaia.parallax_pmra_corr as parallax_pmra_corr,
                gaia.parallax_pmdec_corr as parallax_pmdec_corr,
                gaia.phot_g_mean_mag as phot_g_mean_mag,
                gaia.phot_bp_mean_mag as phot_bp_mean_mag,
                gaia.phot_rp_mean_mag as phot_rp_mean_mag,
                tmass.j_m,
                tmass.h_m,
                tmass.ks_m
            FROM gaiadr3.gaia_source AS gaia
            JOIN gaiaedr3.tmass_psc_xsc_best_neighbour
            AS xmatch USING (source_id)
            JOIN gaiaedr3.tmass_psc_xsc_join
            AS xjoin USING (clean_tmass_psc_xsc_oid)
            JOIN gaiadr1.tmass_original_valid AS tmass
            ON xjoin.original_psc_source_id = tmass.designation
            WHERE 1 = CONTAINS(
                POINT({self.ra}, {self.dec}),
                CIRCLE(gaia.ra, gaia.dec, {cone_radius}))"""
            )
            try:
                cone_objects = job.get_results().to_pandas()
                self.cone_tmass_cross = cone_objects
            except exceptions.HTTPError as hperr:
                print("Received exceptions.HTTPError", hperr)

    def cone_gaia_objects(self, cone_radius):
        """Cone around the target star and colour transform G-Band to K_S-Band.

        Args:
            cone_radius (float): Search cone radius in degree.
        """
        if self.object_found:
            job = Gaia.launch_job_async(
                f"""SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
            FROM gaiadr3.gaia_source AS gaia
            WHERE 1 = CONTAINS(
                POINT({self.ra}, {self.dec}),
                CIRCLE(gaia.ra, gaia.dec, {cone_radius}))"""
            )
            try:
                cone_objects = job.get_results().to_pandas()
                cone_objects["phot_bp_rp_mean_mag"] = (
                    cone_objects["phot_bp_mean_mag"] - cone_objects["phot_rp_mean_mag"]
                )
                cone_objects["ks_m_calc"] = helperfunctions.color_trafo_2MASS_K_S(
                    cone_objects["phot_g_mean_mag"], cone_objects["phot_bp_rp_mean_mag"]
                )
                cone_objects["h_m_calc"] = helperfunctions.color_trafo_2MASS_H(
                    cone_objects["phot_g_mean_mag"], cone_objects["phot_bp_rp_mean_mag"]
                )
                cone_objects["j_m_calc"] = helperfunctions.color_trafo_2MASS_J(
                    cone_objects["phot_g_mean_mag"], cone_objects["phot_bp_rp_mean_mag"]
                )
                self.cone_gaia = cone_objects

            except exceptions.HTTPError as hperr:
                print("Received exceptions.HTTPError", hperr)

    def binning_parameters(self, df, x_col_name, y_col_name, binsize, band):
        """Gaussian 2D fits to each bin.

        Args:
            df (pandas.DataFrame): Cone catalouge data.
            x_col_name (str): pmra, pmdec or parallax.
            y_col_name (str): pmra, pmdec or parallax.
            binsize (float): Number of objects in each single bin.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.
                    'h_m_calc' for Gaia or 'h_m' for 2MASS.
                    'j_m_calc' for Gaia or 'j_m' for 2MASS.

        Returns:
            pandas.DataFrame: Parameters for each bin and each correlation.
        """
        nofbins = int(len(df[band].dropna()) / binsize)

        df_dropna = df[[x_col_name, y_col_name, band]].dropna()
        df_sorted = df_dropna[band].sort_values()
        x_stds, y_stds, x_means, y_means, magnitudes, covs, rhos = (
            [] for i in range(7)
        )
        for samplesize in range(1, nofbins + 1):
            bin_width = samplesize * binsize
            m_min = df_sorted.iloc[bin_width - binsize : bin_width].min()
            m_max = df_sorted.iloc[bin_width - binsize : bin_width].max()
            df_r = df_dropna[df_dropna[band].between(m_min, m_max)]
            if len(df_r) > 1:
                x_data, y_data = helperfunctions.convert_df_to_array(
                    df_r, x_col_name, y_col_name
                )
                (
                    amplitude,
                    x_mean,
                    y_mean,
                    x_stddev,
                    y_stddev,
                    cov,
                    rho,
                ) = helperfunctions.get_g2d_parameters(x_data, y_data)
                x_stds.append(x_stddev)
                y_stds.append(y_stddev)
                x_means.append(x_mean)
                y_means.append(y_mean)
                covs.append(cov)
                rhos.append(rho)
                magnitudes.append((m_max + m_min) / 2)
        catalogue_bin_parameters = pd.DataFrame(
            {
                f"{x_col_name}_mean": x_means,
                f"{y_col_name}_mean": y_means,
                f"{x_col_name}_stddev": x_stds,
                f"{y_col_name}_stddev": y_stds,
                f"cov_{x_col_name}_{y_col_name}": covs,
                "band": magnitudes,
                f"rho_{x_col_name}_{y_col_name}": rhos,
            }
        )
        return catalogue_bin_parameters

    def concat_binning_parameters(self, df_catalogue, band, binsize):
        """Concat the binning parameters of the combinations of pmra,
        pmdec, parallax.

        Args:
            df_catalogue (pandas.DataFrame): Cone catalouge data.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.

        Returns:
            pandas.DataFrame: Binning parameters of different catalogues and variables parameters in a single dataframe.
        """
        catalogue_bp = {}
        for combination in list(
            itertools.combinations(["pmra", "pmdec", "parallax"], r=2)
        ):
            catalogue_bp[
                f"{combination[0]}_{combination[1]}"
            ] = self.binning_parameters(
                df_catalogue,
                combination[0],
                combination[1],
                binsize=binsize,  # int(df_catalogue.shape[0] / 40),
                band=band,
            )
        df_catalogue_bp = pd.concat(
            [catalogue_bp[key] for key in catalogue_bp.keys()], axis=1
        )
        df_catalogue_bp = df_catalogue_bp.loc[
            :, ~df_catalogue_bp.columns.duplicated()
        ].copy()
        self.__setattr__(f"binning_parameters_table_{band}", df_catalogue_bp)
        return df_catalogue_bp

    def calc_background_model_parameters(
        self, list_of_df_bp, band, candidates_df, include_candidates
    ):
        """Fit the binning parameters. For each catalogue and variable there are coeff and cov attributes.
            The syntax:
                variable _ mean or stddev _ coeff or cov _ catalogue name
            The coeff attributes contain the fitting coefficients:
                len(coeff)=3: Fitted with helperfunctions.func_exp.
                len(coeff)=2: Fitted with helperfunctions.func_lin.
                len(coeff)=1: Fitted with helperfunctions.const.


        Args:
            list_of_df_bp (list of pandas.DataFrame s):
            Binned 2D Gaussian Parameters.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.
            candidates_df (pandas.DataFrame): Data on all candidates
            of this host star.
            include_candidates (Boolean): Including the data of the caniddates
              in the fitting.
        """
        df_label = ["gaiacalc", "tmass", "gaiacalctmass"]
        concated_data = pd.concat(list_of_df_bp)
        for idx, df in enumerate([*list_of_df_bp, concated_data]):
            for y_option_value in list(
                itertools.product(["mean", "stddev"], ["pmra", "pmdec", "parallax"])
            ):
                y_option = y_option_value[0]
                pm_value = y_option_value[1]
                #  Fitting with robust regression
                lower_clip = np.percentile(df[band], q=10)
                higher_clip = np.percentile(df[band], q=90)
                #  Shorten the data
                df_clipped = df[df[band].between(lower_clip, higher_clip)]
                data_g2m = df_clipped[[band, f"{pm_value}_{y_option}"]].dropna()
                x_data = data_g2m[band].values
                y_data = data_g2m[f"{pm_value}_{y_option}"].values
                if include_candidates:
                    np.append(x_data, candidates_df[band].values)
                    np.append(y_data, candidates_df[pm_value + "_abs"].values)
                if y_option == "mean" and pm_value in ["pmra", "pmdec"]:
                    fitting_func = helperfunctions.func_lin
                    boundaries = ([-np.inf, -np.inf], [np.inf, np.inf])
                elif y_option == "stddev" and pm_value in ["pmra", "pmdec"]:
                    fitting_func = helperfunctions.func_lin
                    boundaries = (
                        [-np.inf, -np.inf],
                        [np.inf, np.inf],
                    )  # ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                elif pm_value == "parallax" and y_option == "mean":
                    fitting_func = helperfunctions.func_const
                    boundaries = ([-np.inf], [np.inf])
                elif pm_value == "parallax" and y_option == "stddev":
                    fitting_func = helperfunctions.func_const
                    boundaries = ([-np.inf], [np.inf])
                try:
                    popt, pcov = optimize.curve_fit(
                        fitting_func, x_data, y_data, bounds=boundaries
                    )
                    attr_name = f"{pm_value}_{y_option}_model_coeff_{df_label[idx]}"
                    setattr(self, attr_name, popt)
                    attr_name = f"{pm_value}_{y_option}_model_cov_{df_label[idx]}"
                    setattr(self, attr_name, pcov)
                except (RuntimeError, optimize.OptimizeWarning):
                    if y_option == "stddev":
                        try:
                            popt, pcov = optimize.curve_fit(
                                helperfunctions.func_lin, x_data, y_data
                            )
                            attr_name = (
                                f"{pm_value}_{y_option}_model_coeff_{df_label[idx]}"
                            )
                            setattr(self, attr_name, popt)
                            attr_name = (
                                f"{pm_value}_{y_option}_model_cov_{df_label[idx]}"
                            )
                            setattr(self, attr_name, pcov)
                        except (RuntimeError, optimize.OptimizeWarning):
                            print("Fitting error", y_option, pm_value)
            for col in df.columns:
                if "rho" in col:
                    attr_name = f"{col[4:]}_model_{df_label[idx]}"
                    setattr(self, attr_name, df[col].mean())

    def evaluate_candidates_table(self, candidates_df, sigma_model_min, sigma_cc_min):
        """
        Returns all the candidates data of this host star for both catalogues.

        Args:
            candidates_df (pandas.DataFrame): Data on all candidates of this host star.
            sigma_model_min (float or int): The inflating factor for the model likelihood.
            sigma_cc_min (float or int): The inflating factor for its likelihood.
        """
        (
            final_uuids,
            r_tcb_2Dnmodel,
            r_tcb_pmmodel,
            catalogues,
            mean_background_objects,
            mean_measured_positions,
            mean_true_companion,
            cov_background_objects,
            cov_measured_positions,
            cov_true_companion,
            candidate_objects,
            ref_epochs,
        ) = ([] for i in range(12))
        for index_candidate in range(len(candidates_df)):
            for catalogue in ["tmass", "gaiacalctmass"]:
                #  Create candidate object
                candidate = Candidate(
                    candidates_df,
                    index_candidate=index_candidate,
                    host_star=self,
                    band="band",
                    catalogue=catalogue,
                )
                epochs = list(
                    np.around(
                        candidate.cc_true_data["t_days_since_Gaia"] / 365.25
                        + self.ref_epoch,
                        1,
                    )
                )
                # Compute liklihoods
                candidate.calc_likelihoods_2Dnmodel(self)
                candidate.calc_likelihoods_pmmodel(self, sigma_model_min, sigma_cc_min)
                # Compute odds ratio
                candidate.calc_prob_ratio_pmmodel()
                candidate.calc_prob_ratio_2Dnmodel()
                final_uuids.append(candidate.final_uuid)
                r_tcb_2Dnmodel.append(candidate.r_tcb_2Dnmodel)
                r_tcb_pmmodel.append(candidate.r_tcb_pmmodel)
                catalogues.append(catalogue)
                mean_background_objects.append(candidate.mean_background_object)
                mean_measured_positions.append(candidate.mean_measured_positions)
                mean_true_companion.append(candidate.mean_true_companion)
                cov_background_objects.append(candidate.cov_background_object)
                cov_measured_positions.append(candidate.cov_measured_positions)
                cov_true_companion.append(candidate.cov_true_companion)
                ref_epochs.append(epochs)
            candidate_objects.append(candidate)
        candidates_r_tcb = pd.DataFrame(
            {
                "final_uuid": final_uuids,
                "mean_background_object": mean_background_objects,
                "mean_measured_positions": mean_measured_positions,
                "mean_true_companion": mean_true_companion,
                "cov_background_object": cov_background_objects,
                "cov_measured_positions": cov_measured_positions,
                "cov_true_companion": cov_true_companion,
                "r_tcb_2Dnmodel": r_tcb_2Dnmodel,
                "r_tcb_pmmodel": r_tcb_pmmodel,
                "r_tcb_catalogue": catalogues,
                "ref_epochs": ref_epochs,
            }
        )
        candidates = candidates_df.merge(candidates_r_tcb, on=["final_uuid"])
        self.candidates = candidates
        self.candidates_objects = candidate_objects


class BackgroundModel:
    def __init__(self, candidate_mag, host_star_object, catalogue_name):
        self.pmra = (
            host_star_object.__getattribute__(
                f"pmra_mean_model_coeff_{catalogue_name}"
            )[0]
            * candidate_mag
            + host_star_object.__getattribute__(
                f"pmra_mean_model_coeff_{catalogue_name}"
            )[1]
        )

        if (
            len(
                host_star_object.__getattribute__(
                    f"pmra_stddev_model_coeff_{catalogue_name}"
                )
            )
            == 3
        ):
            self.pmra_error = (
                host_star_object.__getattribute__(
                    f"pmra_stddev_model_coeff_{catalogue_name}"
                )[0]
                * np.exp(
                    -host_star_object.__getattribute__(
                        f"pmra_stddev_model_coeff_{catalogue_name}"
                    )[1]
                    * candidate_mag
                )
                + host_star_object.__getattribute__(
                    f"pmra_stddev_model_coeff_{catalogue_name}"
                )[2]
            )
        else:
            self.pmra_error = (
                host_star_object.__getattribute__(
                    f"pmra_stddev_model_coeff_{catalogue_name}"
                )[0]
                * candidate_mag
                + host_star_object.__getattribute__(
                    f"pmra_stddev_model_coeff_{catalogue_name}"
                )[1]
            )
        self.pmdec = (
            host_star_object.__getattribute__(
                f"pmdec_mean_model_coeff_{catalogue_name}"
            )[0]
            * candidate_mag
            + host_star_object.__getattribute__(
                f"pmdec_mean_model_coeff_{catalogue_name}"
            )[1]
        )
        if (
            len(
                host_star_object.__getattribute__(
                    f"pmdec_stddev_model_coeff_{catalogue_name}"
                )
            )
            == 3
        ):
            self.pmdec_error = (
                host_star_object.__getattribute__(
                    f"pmdec_stddev_model_coeff_{catalogue_name}"
                )[0]
                * np.exp(
                    -host_star_object.__getattribute__(
                        f"pmdec_stddev_model_coeff_{catalogue_name}"
                    )[1]
                    * candidate_mag
                )
                + host_star_object.__getattribute__(
                    f"pmdec_stddev_model_coeff_{catalogue_name}"
                )[2]
            )
        else:
            self.pmdec_error = (
                host_star_object.__getattribute__(
                    f"pmdec_stddev_model_coeff_{catalogue_name}"
                )[0]
                * candidate_mag
                + host_star_object.__getattribute__(
                    f"pmdec_stddev_model_coeff_{catalogue_name}"
                )[1]
            )

        self.pmra_pmdec_corr = host_star_object.__getattribute__(
            f"pmra_pmdec_model_{catalogue_name}"
        )

        self.parallax = host_star_object.__getattribute__(
            f"parallax_mean_model_coeff_{catalogue_name}"
        )[0]

        self.parallax_error = host_star_object.__getattribute__(
            f"parallax_stddev_model_coeff_{catalogue_name}"
        )[0]

        self.parallax_pmra_corr = host_star_object.__getattribute__(
            f"pmra_parallax_model_{catalogue_name}"
        )
        self.parallax_pmdec_corr = host_star_object.__getattribute__(
            f"pmdec_parallax_model_{catalogue_name}"
        )


class Survey:
    """Creates odds ratio table based on the observational data of candidates
    and the field star models."""

    def __init__(self, survey, survey_bandfilter_colname):
        """
        Based on the target name returns a dataframe
        containing all the survey data to this target
        and calculates the proper motion.

        Args:
            survey (pandas.DataFrame): Contains survey data. Necessary columns are:\n
                            - Main_ID: Host star name.\n
                            - final_uuid: Unique identifier of the two measurements of the same candidate.\n
                            - dRA: Relative distance candidate-hoststar in mas.\n
                            - dRA_err: Respective error.\n
                            - dDEC: Relative distance candidate-hoststar in mas.\n
                            - dDEC_err: Respective error.\n
                            - mag0: Magnitude of the candidate.\n
            survey_bandfilter_colname (str): Column name of survey magnitudes, e.g. 'mag0'

        Returns:
            pandas.DataFrame: Contains the filtered survey data.
        """
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s", "%H:%M"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        logger.info("Preprocessing data starts...")
        targets = []
        for target_name in tqdm.tqdm(
            survey["Main_ID"].unique(),
            desc="Preparing candidates",
            ncols=100,
            colour="green",
            ascii=" 123456789#",
        ):
            survey_target = survey[survey["Main_ID"] == target_name]
            host_star = HostStar(target=target_name)
            if (
                survey_target["final_uuid"].value_counts() < 2
            ).all() or host_star.object_found is False:
                logger.info(
                    f"{target_name}: Candidates were once observed or final_uuid is only once in the data or star not found."
                )
            else:
                targets.append(target_name)
                (
                    final_uuids,
                    mean_pmras,
                    mean_pmdecs,
                    band,
                    band_error,
                    pmra_error,
                    pmdec_error,
                    pmra_pmdec_corr,
                    dRAs,
                    dDECs,
                    times,
                    dRAs_err,
                    dDECs_err,
                    dRA_dDEC_corrs,
                    seps,
                    ts_days_since_Gaia,
                ) = ([] for i in range(16))
                survey_target = survey[survey["Main_ID"] == target_name].copy()
                survey_target["date"] = pd.to_datetime(survey_target["date"])
                survey_target.loc[:, "t_day_since_Gaia"] = (
                    pd.to_datetime(survey_target["date"]).apply(
                        lambda x: x.to_julian_date()
                    )
                    - 2451545.0
                    - (host_star.ref_epoch - 2000) * 365.25
                ).astype(int)
                for final_uuid in survey_target["final_uuid"].unique():
                    survey_finaluuid = survey_target[
                        survey_target["final_uuid"] == final_uuid
                    ]
                    if len(survey_finaluuid) >= 2:
                        time = survey_finaluuid["date"].values
                        dRA_dDEC = survey_finaluuid[["dRA", "dDEC"]].values
                        dRA_dDEC_err = survey_finaluuid[["dRA_err", "dDEC_err"]].values
                        deltayears = (time[-1] - time[0]).astype(
                            "timedelta64[D]"
                        ).astype("int") / 365.25
                        pmra, pmdec = (dRA_dDEC[-1] - dRA_dDEC[0]) / deltayears
                        bst_err1 = np.sqrt(
                            (dRA_dDEC_err[-1] / deltayears) ** 2
                            + (dRA_dDEC_err[0] / deltayears) ** 2
                        )
                        pmra_err, pmdec_err = bst_err1[0], bst_err1[1]
                        sep = (
                            np.mean(survey_finaluuid["dRA"].values) ** 2
                            + np.mean(survey_finaluuid["dDEC"].values) ** 2
                        ) ** (1 / 2)
                        times.append(time)
                        dRAs.append(survey_finaluuid["dRA"].values)
                        dDECs.append(survey_finaluuid["dDEC"].values)
                        dRAs_err.append(survey_finaluuid["dRA_err"].values)
                        dDECs_err.append(survey_finaluuid["dDEC_err"].values)
                        dRA_dDEC_corrs.append(survey_finaluuid["dRA_dDEC_corr"].values)
                        mean_pmras.append(pmra)
                        mean_pmdecs.append(pmdec)
                        pmra_pmdec_corr.append(0)
                        pmra_error.append(pmra_err)
                        pmdec_error.append(pmdec_err)
                        final_uuids.append(final_uuid)
                        seps.append(sep)
                        band.append(survey_finaluuid[survey_bandfilter_colname].mean())
                        band_error.append(
                            survey_finaluuid[f"{survey_bandfilter_colname}_err"].mean()
                        )
                        ts_days_since_Gaia.append(
                            survey_finaluuid["t_day_since_Gaia"].values
                        )
                df_survey = pd.DataFrame(
                    {
                        "final_uuid": final_uuids,
                        "dates": times,
                        "dRA": dRAs,
                        "dDEC": dDECs,
                        "dRA_err": dRAs_err,
                        "dDEC_err": dDECs_err,
                        "dRA_dDEC_corr": dRA_dDEC_corrs,
                        "pmra_mean": mean_pmras,
                        "pmdec_mean": mean_pmdecs,
                        "pmra_pmdec_corr": pmra_pmdec_corr,
                        "band": band,
                        "band_error": band_error,
                        "pmra_error": pmra_error,
                        "pmdec_error": pmdec_error,
                        "sep": seps,
                        "t_days_since_Gaia": ts_days_since_Gaia,
                    }
                )
                df_survey["pmra_abs"] = df_survey["pmra_mean"] + host_star.pmra
                df_survey["pmdec_abs"] = df_survey["pmdec_mean"] + host_star.pmdec

                df_survey["pmra_abs_error"] = (
                    df_survey["pmra_error"] ** 2 + host_star.pmra_error**2
                ) ** (1 / 2)
                df_survey["pmdec_abs_error"] = (
                    df_survey["pmdec_error"] ** 2 + host_star.pmdec_error**2
                ) ** (1 / 2)
                self.__setattr__(f"candidates_data_{target_name}", df_survey)
        self.target_names = targets

    def set_fieldstar_models(
        self, binning_band_trafo, binning_band, cone_radius=0.3, binsize=200
    ):
        """Build for each host star given in survey a field star model.

        Args:
            binning_band_trafo (str): Color transformed bandwidth, e.g. ks_m_calc.
            binning_band (str): Color bandwidth, e.g. ks_m.
            cone_radius (float): Default=0.3 in degrees.
            binsize (int): Default=200 in number of field stars per magnitude bin.
        """
        for target_name in tqdm.tqdm(
            self.target_names,
            desc="Building models",
            ncols=100,
            colour="green",
            ascii=" 123456789#",
        ):
            #  Query host star data
            host_star = HostStar(target_name)
            #  Query cone data (Gaia and 2MASS)
            #  Dataframe variables
            host_star.cone_tmasscross_objects(cone_radius)
            host_star.cone_gaia_objects(cone_radius)
            df_gaia = host_star.cone_gaia
            df_tmass = host_star.cone_tmass_cross
            #  Gaia data without 2MASS
            df_gaia_without_tmass = df_gaia[
                ~df_gaia.source_id.isin(df_tmass.source_id.to_list())
            ]
            #  Binning parameters for 2MASS and Gaia
            #  Merge the binning parameters to a gaia df and tmass df
            #  And drop the duplicated columns
            #  Binning parameters
            df_gaia_bp = host_star.concat_binning_parameters(
                df_gaia_without_tmass, binning_band_trafo, binsize
            )
            df_tmass_bp = host_star.concat_binning_parameters(
                df_tmass, binning_band, binsize
            )
            #  Calculate the fit coefficients
            #  Can be accessed e.g. host_star.parallax_mean_model_coeff_gaiacalc
            host_star.calc_background_model_parameters(
                list_of_df_bp=[df_gaia_bp, df_tmass_bp],
                band="band",
                candidates_df=None,
                include_candidates=False,
            )
            self.__setattr__(f"fieldstar_model_{target_name}", host_star)

    def set_evaluated_fieldstar_models(self, sigma_cc_min, sigma_model_min):
        """Evaluate the field star models with candidate data.

        Args:
            sigma_cc_min (float): Minimum sigma for the candidates likelihoods.
            sigma_model_min (float): Minimum sigma for the field star model likelihoods.
        """
        for target_name in tqdm.tqdm(
            [el[16:] for el in list(self.__dict__) if "candidates_data" in el],
            desc="Evaluate candidates",
            ncols=100,
            colour="green",
            ascii=" 123456789#",
        ):
            # Load the model for target_name
            field_star_model = self.__getattribute__(f"fieldstar_model_{target_name}")
            # Evaluate the model for target_name
            field_star_model.evaluate_candidates_table(
                self.__getattribute__(f"candidates_data_{target_name}"),
                sigma_cc_min=sigma_cc_min,
                sigma_model_min=sigma_model_min,
            )
            # Write the results
            candidates_table = self.__getattribute__(
                f"fieldstar_model_{target_name}"
            ).candidates
            self.__setattr__(f"fieldstar_model_results_{target_name}", candidates_table)

    def get_true_companions(self, threshold=0):
        """Return all candidates with a odds ratio greater than the threshold.

        Args:
            threshold (float): Odds ratio greater than the threshold will be returned.

        Returns:
            pandas.DataFrame: Candidates with r_tcb>threshold.
        """
        candidate_results = []
        target_names = [
            el[24:] for el in list(self.__dict__) if "fieldstar_model_results_" in el
        ]
        for target_name in target_names:
            candidates_table = self.__getattribute__(
                f"fieldstar_model_results_{target_name}"
            )
            candidates_table["target_name"] = target_name
            candidates_table = candidates_table[
                candidates_table.r_tcb_2Dnmodel > threshold
            ]
            candidate_results.append(candidates_table)
        candidates_table_th = pd.concat(candidate_results).reset_index()
        return candidates_table_th
