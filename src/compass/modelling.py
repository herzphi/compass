from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from requests.exceptions import HTTPError
from astropy.modeling.functional_models import Gaussian2D, Gaussian1D
import itertools
import re
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from scipy.special import expit
from scipy.optimize import curve_fit, OptimizeWarning


def get_p_ratio_table(target, cone_radius, candidates_raw_data, sigma_cc_min, sigma_model_min):
    """Calculates the odds ratio based on the target,
    the model and the raw candidates data.
    Args:
        target (str): Name of the host star.
        cone_radius (float): Radius of the cone in degrees.
        candidates_raw_data (pandas.DataFrame): Data of the candidates. 
            Column names can be adjusted in the candidate class.
        sigma_cc_min (float): Inflating the candidate likelihood in mas/yr.
        sigma_model_min (float): Inflating the model likelihood in mas/yr.
    Returns:
        candidates_table (pandas.DataFrame): Contains candidates_raw_data and the p_ratios.
    """
    #  Query host star data
    host_star = HostStar(target)
    #  Query cone data (Gaia and 2MASS)
    #  Dataframe variables
    host_star.cone_tmasscross_objects(cone_radius)
    host_star.cone_gaia_objects(cone_radius)
    df_gaia = host_star.cone_gaia
    df_tmass = host_star.cone_tmass_cross
    #  Gaia data without 2MASS
    df_gaia_without_tmass = df_gaia[~df_gaia.source_id.isin(df_tmass.source_id.to_list())]
    #  Binning parameters for 2MASS and Gaia
    #  Merge the binning parameters to a gaia df and tmass df
    df_gaia_bp = host_star.concat_binning_parameters(df_gaia_without_tmass, 'ks_m_calc')
    df_tmass_bp = host_star.concat_binning_parameters(df_tmass, 'ks_m')
    #  Calculate the fit coefficients
    #  Can be accessed e.g. host_star.parallax_mean_model_coeff_gaiacalc
    host_star.pmm_parameters(
        list_of_df_bp=[df_gaia_bp,df_tmass_bp],
        band='band',
        candidates_df=None,
        include_candidates=False
    )
    #  Get the candidates data
    candidates_calc_df = get_candidates_parameters(target, candidates_raw_data, host_star)
    #  Calculate p_ratios
    host_star.candidates_table(
        candidates_calc_df,
        sigma_cc_min=sigma_cc_min,
        sigma_model_min=sigma_model_min
    )
    candidates_table = host_star.candidates
    return candidates_table


def get_candidates_parameters(target, survey, host_star):
    """
    Based on the target name returns a dataframe
    containing all the survey data to this target
    and calculates the proper motion.
    Args:
        target (str): Name of the host star.
        survey (pandas.DataFrame): Contains survey data.
    Returns:
        df_survey (pandas.DataFrame): Contains the filtered survey data.
    """
    survey_target = survey[survey['Main_ID'] == target]
    if len(survey_target['final_uuid'].unique()) < 2:
        raise ValueError('There are not enough cross matches in the survey.')
    if len(survey_target['date'].unique()) < 2:
        raise ValueError('This object was once observed. Necessary are at least two observations.')

    final_uuids, mean_pmras, mean_pmdecs, band, \
        band_error, pmra_error, pmdec_error, snr_list,\
        sep, sep_mean, pmra_pmdec_corr, dRAs, dDECs, times, dRAs_err, dDECs_err = ([] for i in range(16))
    survey_target = survey[survey['Main_ID'] == target].copy()
    survey_target['date'] = pd.to_datetime(survey_target['date'])

    for final_uuid in survey_target['final_uuid'].unique():
        survey_finaluuid = survey_target[survey_target['final_uuid'] == final_uuid]
        if len(survey_finaluuid) >= 2:
            time = survey_finaluuid['date'].values
            dRA_dDEC = survey_finaluuid[['dRA', 'dDEC']].values
            dRA_dDEC_err = survey_finaluuid[['dRA_err', 'dDEC_err']].values
            deltayears = (time[1] - time[0]).astype('timedelta64[D]').astype('int')/365.25
            pmra, pmdec = ((dRA_dDEC[1] - dRA_dDEC[0])/deltayears)
            bst_err1 = np.sqrt((dRA_dDEC_err[1]/deltayears)**2 + (dRA_dDEC_err[0]/deltayears)**2)
            pmra_err, pmdec_err = bst_err1[0], bst_err1[1]

            times.append(time)
            dRAs.append(survey_finaluuid['dRA'].values)
            dDECs.append(survey_finaluuid['dDEC'].values)
            dRAs_err.append(survey_finaluuid['dRA_err'].values)
            dDECs_err.append(survey_finaluuid['dDEC_err'].values)
            snr_list.append(survey_finaluuid['snr0'].mean())
            mean_pmras.append(pmra)
            mean_pmdecs.append(pmdec)
            pmra_pmdec_corr.append(0)
            pmra_error.append(pmra_err)
            pmdec_error.append(pmdec_err)
            final_uuids.append(final_uuid)
            band.append(survey_finaluuid['mag0'].mean())
            band_error.append(survey_finaluuid['mag0_err'].mean())
            sep.append(survey_finaluuid['sep'].values)
            sep_mean.append(np.mean(survey_finaluuid['sep'].values))
    df_survey = pd.DataFrame({
            'final_uuid': final_uuids,
            'dates': times,
            'dRA': dRAs,
            'dDEC': dDECs,
            'dRA_err': dRAs_err,
            'dDEC_err': dDECs_err,
            'pmra_mean': mean_pmras,
            'pmdec_mean': mean_pmdecs,
            'pmra_pmdec_corr': pmra_pmdec_corr,
            'band': band,
            'band_error': band_error,
            'pmra_error': pmra_error,
            'pmdec_error': pmdec_error,
            'snr_list': snr_list,
            'sep': sep,
            'sep_mean': sep_mean
    })
    df_survey['pmra_abs'] = df_survey['pmra_mean']+host_star.pmra
    df_survey['pmdec_abs'] = df_survey['pmdec_mean']+host_star.pmdec

    df_survey['pmra_abs_error'] = (df_survey['pmra_error']**2+host_star.pmra_error**2)**(1/2)
    df_survey['pmdec_abs_error'] = (df_survey['pmdec_error']**2+host_star.pmdec_error**2)**(1/2)
    return df_survey


class CovarianceMatrix():
    def calc_variance_x(time, plx_proj, host_star):
        cov_plx_pm = host_star.pmra_error*host_star.parallax_error*host_star.parallax_pmra_corr
        var_prime = (time*host_star.pmra_error)**2+(plx_proj*host_star.parallax_error)**2+2*time*plx_proj*cov_plx_pm
        return var_prime

    def calc_variance_y(time, plx_proj, host_star):
        cov_plx_pm = host_star.pmdec_error*host_star.parallax_error*host_star.parallax_pmdec_corr
        var_prime = (time*host_star.pmdec_error)**2+(plx_proj*host_star.parallax_error)**2+2*time*plx_proj*cov_plx_pm
        return var_prime

    def calc_covariance_xiyi(time, plx_proj_x, plx_proj_y, host_star):
        cov_plx_pmra = host_star.pmra_error*host_star.parallax_error*host_star.parallax_pmra_corr
        cov_plx_pmdec = host_star.pmdec_error*host_star.parallax_error*host_star.parallax_pmdec_corr
        cov = time**2 * host_star.pmra_pmdec_corr*host_star.pmra_error*host_star.pmdec_error
        + plx_proj_x*plx_proj_y*host_star.parallax_error**2
        + time*plx_proj_y*cov_plx_pmra
        + time*plx_proj_x*cov_plx_pmdec
        return cov

    def calc_covariance_xixj(timei, timej, plx_proj_ra_i, plx_proj_ra_j, host_star):
        cov_plx_pmra = host_star.pmra_error*host_star.parallax_error*host_star.parallax_pmra_corr
        cov = timei*timej*host_star.pmra_error**2
        + plx_proj_ra_i*plx_proj_ra_j*host_star.parallax_error**2
        + (timei*plx_proj_ra_j + timej*plx_proj_ra_i)*cov_plx_pmra
        return cov

    def calc_covariance_yiyj(timei, timej, plx_proj_dec_i, plx_proj_dec_j, host_star):
        cov_plx_pmdec = host_star.pmdec_error*host_star.parallax_error*host_star.parallax_pmdec_corr
        cov = timei*timej*host_star.pmdec_error**2
        + plx_proj_dec_i*plx_proj_dec_j*host_star.parallax_error**2
        + (timei*plx_proj_dec_j + timej*plx_proj_dec_i)*cov_plx_pmdec
        return cov

    def calc_covariance_xiyj(timei, timej, plx_proj_ra_i, plx_proj_dec_j, host_star):
        cov_pmra_pmdec = host_star.pmra_error*host_star.pmdec_error*host_star.pmra_pmdec_corr
        cov_plx_pmra = host_star.pmra_error*host_star.parallax_error*host_star.parallax_pmra_corr
        cov_plx_pmdec = host_star.pmdec_error*host_star.parallax_error*host_star.parallax_pmdec_corr
        cov = timei*timej*cov_pmra_pmdec
        + plx_proj_ra_i*plx_proj_dec_j*host_star.parallax_error**2
        + timei*plx_proj_dec_j*cov_plx_pmra
        + timej*plx_proj_ra_i*cov_plx_pmdec
        return cov

    def calc_covariance_yixj(timei, timej, plx_proj_dec_i, plx_proj_ra_j, host_star):
        cov_pmra_pmdec = host_star.pmra_error*host_star.pmdec_error*host_star.pmra_pmdec_corr
        cov_plx_pmra = host_star.pmra_error*host_star.parallax_error*host_star.parallax_pmra_corr
        cov_plx_pmdec = host_star.pmdec_error*host_star.parallax_error*host_star.parallax_pmdec_corr
        cov = timei*timej*cov_pmra_pmdec
        + plx_proj_dec_i*plx_proj_ra_j*host_star.parallax_error**2
        + timej*plx_proj_dec_i*cov_plx_pmra+timei*plx_proj_ra_j*cov_plx_pmdec
        return cov

    def cov_propagation(C_0, days_1, days_2, plx_proj_x, plx_proj_y, host_star):
        time = (days_2-days_1)/365.35
        var_x_prime_2 = CovarianceMatrix.calc_variance_x(time, plx_proj_x, host_star)
        var_y_prime_2 = CovarianceMatrix.calc_variance_y(time, plx_proj_y, host_star)
        cov_x_prime_2_y_prime_2 = CovarianceMatrix.calc_covariance_xiyi(
            time,
            plx_proj_x,
            plx_proj_y,
            host_star
        )
        C_prime_2 = np.array([
            [var_x_prime_2, cov_x_prime_2_y_prime_2],
            [cov_x_prime_2_y_prime_2, var_y_prime_2]]
        )
        C = C_0 + C_prime_2
        return C

    def covariance_matrix(times, plx_proj_ra, plx_proj_dec, host_star):
        dict_var_covar = {}
        J = list(range(1, len(times)+1))
        for i, time in enumerate(times):
            dict_var_covar[f'var_x{i+1}'] = CovarianceMatrix.calc_variance_x(time/365.25, plx_proj_ra[time], host_star)
            dict_var_covar[f'var_y{i+1}'] = CovarianceMatrix.calc_variance_y(time/365.25, plx_proj_dec[time], host_star)
            for j in J:
                if i+1 != j:
                    dict_var_covar[f'covar_x{i+1}x{j}'] = CovarianceMatrix.calc_covariance_xixj(times[i]/365.25, times[j-1]/365.25, plx_proj_ra[times[i]], plx_proj_ra[times[j-1]], host_star)
                    dict_var_covar[f'covar_y{i+1}y{j}'] = CovarianceMatrix.calc_covariance_yiyj(times[i]/365.25, times[j-1]/365.25, plx_proj_dec[times[i]], plx_proj_dec[times[j-1]], host_star)
                dict_var_covar[f'covar_x{i+1}y{j}'] = CovarianceMatrix.calc_covariance_xiyi(times[j-1]/365.25, plx_proj_ra[times[j-1]], plx_proj_dec[times[j-1]], host_star)
                if f'covar_x{i+1}y{j}' in dict_var_covar.keys() and i+1==j:
                    continue
                else:
                    dict_var_covar[f'covar_x{i+1}y{j}'] = CovarianceMatrix.calc_covariance_xiyj(times[i]/365.25, times[j-1]/365.25, plx_proj_ra[times[i]], plx_proj_dec[times[j-1]], host_star)
                    dict_var_covar[f'covar_y{i+1}x{j}'] = CovarianceMatrix.calc_covariance_yixj(times[i]/365.25, times[j-1]/365.25, plx_proj_dec[times[i]], plx_proj_ra[times[j-1]], host_star)
            J = J[1:]
        var_keys = [el for el in list(dict_var_covar.keys()) if el.startswith('var')]
        empty_matrix = np.zeros((len(var_keys), len(var_keys)))
        np.fill_diagonal(empty_matrix, np.array([dict_var_covar[var_key] for var_key in var_keys]))
        # COVARIANCES INTO MATRIX
        sorted_covars = {}
        for i in range(len(times)):
            sorted_covars[f'covar_keys_x{i+1}'] = sorted([el for el in list(dict_var_covar.keys()) if el.startswith(f'covar_x{i+1}')], key=lambda x: x[-1:])
            sorted_covars[f'covar_keys_y{i+1}'] = sorted([el for el in list(dict_var_covar.keys()) if el.startswith(f'covar_y{i+1}')], key=lambda x: x[-2:])
        for i, cov_list in enumerate(list(sorted_covars.keys())):
            empty_matrix[i][i+1:] = np.array([dict_var_covar[covar_key] for covar_key in sorted_covars[cov_list]])
            empty_matrix[:, i][i+1:] = np.array([dict_var_covar[covar_key] for covar_key in sorted_covars[cov_list]])
        return empty_matrix


class CovarianceMatrixPopulation():
    def calc_variance_x(plx_proj, host_star):
        var_prime = (plx_proj*host_star.parallax_error)**2
        return var_prime

    def calc_variance_y(plx_proj, host_star):
        var_prime = (plx_proj*host_star.parallax_error)**2
        return var_prime

    def calc_covariance_xiyi(plx_proj_x, plx_proj_y, host_star):
        cov = plx_proj_x*plx_proj_y*host_star.parallax_error**2
        return cov

    def calc_covariance_xixj(plx_proj_ra_i, plx_proj_ra_j, host_star):
        cov = plx_proj_ra_i*plx_proj_ra_j*host_star.parallax_error**2
        return cov

    def calc_covariance_yiyj(plx_proj_dec_i, plx_proj_dec_j, host_star):
        cov = plx_proj_dec_i*plx_proj_dec_j*host_star.parallax_error**2
        return cov

    def calc_covariance_xiyj(plx_proj_ra_i, plx_proj_dec_j, host_star):
        cov = plx_proj_ra_i*plx_proj_dec_j*host_star.parallax_error**2
        return cov

    def calc_covariance_yixj(plx_proj_dec_i, plx_proj_ra_j, host_star):
        cov = plx_proj_dec_i*plx_proj_ra_j*host_star.parallax_error**2
        return cov

    def cov_propagation(C_0, days_1, days_2, plx_proj_x, plx_proj_y, host_star):
        var_x_prime_2 = CovarianceMatrixPopulation.calc_variance_x(plx_proj_x, host_star)
        var_y_prime_2 = CovarianceMatrixPopulation.calc_variance_y(plx_proj_y, host_star)
        cov_x_prime_2_y_prime_2 = CovarianceMatrixPopulation.calc_covariance_xiyi(
            plx_proj_x,
            plx_proj_y,
            host_star
        )
        C_prime_2 = np.array([
            [var_x_prime_2, cov_x_prime_2_y_prime_2],
            [cov_x_prime_2_y_prime_2, var_y_prime_2]]
        )
        C = C_0 + C_prime_2
        return C

    def covariance_matrix(times, plx_proj_ra, plx_proj_dec, host_star):
        dict_var_covar = {}
        J = list(range(1, len(times)+1))
        for i, time in enumerate(times):
            dict_var_covar[f'var_x{i+1}'] = CovarianceMatrixPopulation.calc_variance_x(plx_proj_ra[time], host_star)
            dict_var_covar[f'var_y{i+1}'] = CovarianceMatrixPopulation.calc_variance_y(plx_proj_dec[time], host_star)
            for j in J:
                if i+1 != j:
                    dict_var_covar[f'covar_x{i+1}x{j}'] = CovarianceMatrixPopulation.calc_covariance_xixj(plx_proj_ra[times[i]], plx_proj_ra[times[j-1]], host_star)
                    dict_var_covar[f'covar_y{i+1}y{j}'] = CovarianceMatrixPopulation.calc_covariance_yiyj(plx_proj_dec[times[i]], plx_proj_dec[times[j-1]], host_star)
                dict_var_covar[f'covar_x{i+1}y{j}'] = CovarianceMatrixPopulation.calc_covariance_xiyi(plx_proj_ra[times[j-1]], plx_proj_dec[times[j-1]], host_star)
                if f'covar_x{i+1}y{j}' in dict_var_covar.keys() and i+1==j:
                    continue
                else:
                    dict_var_covar[f'covar_x{i+1}y{j}'] = CovarianceMatrixPopulation.calc_covariance_xiyj(plx_proj_ra[times[i]], plx_proj_dec[times[j-1]], host_star)
                    dict_var_covar[f'covar_y{i+1}x{j}'] = CovarianceMatrixPopulation.calc_covariance_yixj(plx_proj_dec[times[i]], plx_proj_ra[times[j-1]], host_star)
            J = J[1:]
        var_keys = [el for el in list(dict_var_covar.keys()) if el.startswith('var')]
        empty_matrix = np.zeros((len(var_keys), len(var_keys)))
        np.fill_diagonal(empty_matrix, np.array([dict_var_covar[var_key] for var_key in var_keys]))
        # COVARIANCES INTO MATRIX
        sorted_covars = {}
        for i in range(len(times)):
            sorted_covars[f'covar_keys_x{i+1}'] = sorted([el for el in list(dict_var_covar.keys()) if el.startswith(f'covar_x{i+1}')], key=lambda x: x[-1:])
            sorted_covars[f'covar_keys_y{i+1}'] = sorted([el for el in list(dict_var_covar.keys()) if el.startswith(f'covar_y{i+1}')], key=lambda x: x[-2:])
        for i, cov_list in enumerate(list(sorted_covars.keys())):
            empty_matrix[i][i+1:] = np.array([dict_var_covar[covar_key] for covar_key in sorted_covars[cov_list]])
            empty_matrix[:, i][i+1:] = np.array([dict_var_covar[covar_key] for covar_key in sorted_covars[cov_list]])
        return empty_matrix


class HelperFunctions:
    def get_ellipse_props(cov, confidence):
        """
        Calculates the eigenvalues/eigenvectors of the covariance matrix
        and returns the major and minor axis length of the ellipse.
            Args:
                cov (numpy.array (2x2)): Covariance matrix.
                confidence (float): Probability between 0 and 1.
            Returns:
                major_axis (float): Semi-major axis.
                minor_axis (float): Semi-minor axis.
                angle (float): Angle of rotation of the ellipse.
        """
        # Compute the value of the chi-squared distribution
        # that corresponds to the
        # desired level of confidence
        r_prime = np.sqrt(-2*np.log(1-confidence))
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        major_axis = np.sqrt(eigenvalues[0])*r_prime
        minor_axis = np.sqrt(eigenvalues[1])*r_prime
        angle = np.arctan(eigenvectors[1, 0]/eigenvectors[1, 1])
        return major_axis, minor_axis, angle

    def add_ellp_patch(pdf, major, minor, angle, color, axis):
        '''
            Adds a patch to a figure. The patch includes a
            ellipse at the means of a pdf with a scalable
            magnitude in the both axis directions.
            Args:
                pdf (astropy.modeling.Gaussian2D): 2D Gaussian function.
                eigvalue (list): List of eigenvalues of the cov matrix.
                angle (float): Angle of the eigenvectors in radian.
                i (float): Scaling the magnitude of the eigenvalues.
                axis (figure axis): E.g. plt, axs[1,0]
        '''
        axis.add_patch(
                Ellipse(
                    (pdf.x_mean.value, pdf.y_mean.value),
                    width=2*major,
                    height=2*minor,
                    angle=360*angle/(2*np.pi),
                    facecolor='none',
                    edgecolor=color,
                    linewidth=1,
                    # label=f'{i*100:.0f}'
                )
            )

    def add_ellp_patch2(x_mean, y_mean, major, minor, angle, color, linestyle, axis):
        '''
            Adds a patch to a figure. The patch includes a
            ellipse at the means of a pdf with a scalable
            magnitude in the both axis directions.
            Args:
                pdf (astropy.modeling.Gaussian2D): 2D Gaussian function.
                eigvalue (list): List of eigenvalues of the cov matrix.
                angle (float): Angle of the eigenvectors in radian.
                i (float): Scaling the magnitude of the eigenvalues.
                axis (figure axis): E.g. plt, axs[1,0]
        '''
        axis.add_patch(
                Ellipse(
                    (x_mean, y_mean),
                    width=2*major,
                    height=2*minor,
                    angle=360*angle/(2*np.pi),
                    facecolor='none',
                    edgecolor=color,
                    linewidth=1,
                    linestyle=linestyle
                    # label=f'{i*100:.0f}'
                )
            )

    def convert_df_to_array(df, x_col_name, y_col_name):
        '''
            Convert dataframe to two 1D arrays.
        '''
        data = df[[x_col_name, y_col_name]].dropna()
        x_data = data[x_col_name].values
        y_data = data[y_col_name].values
        return x_data, y_data

    def gaussian1D(g2d_astropy_func, x_or_y):
        stddev = f'{x_or_y}_stddev'
        mean = f'{x_or_y}_mean'
        g1d = Gaussian1D(
            (np.sqrt(2*np.pi)*g2d_astropy_func.__getattribute__(stddev))**(-1),
            g2d_astropy_func.__getattribute__(mean),
            g2d_astropy_func.__getattribute__(stddev)
        )
        return g1d

    def get_g2d_parameters(x_data, y_data):
        '''
            Calculate 2D Gaussian parameters based on two arrays.
        '''
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)
        x_stddev = np.std(x_data)
        y_stddev = np.std(y_data)
        cov = np.cov(m=[x_data, y_data])
        rho = cov[1, 0]/(x_stddev*y_stddev)
        amplitude = 2*np.pi*x_stddev*y_stddev*np.sqrt(1-rho**2)
        return amplitude, x_mean, y_mean, x_stddev, y_stddev, cov, rho

    def get_g2d_func(x_mean, y_mean, x_std, y_std, rho):
        cov = np.array([
            [x_std**2, rho*x_std*y_std],
            [rho*x_std*y_std, y_std**2]
        ])
        denom = 2*np.pi*np.linalg.det(cov)**(1/2)
        g2d = Gaussian2D(
            amplitude=1/denom,
            x_mean=x_mean,
            y_mean=y_mean,
            cov_matrix=cov
        )
        return g2d, cov

    def convolution2d(a, b, A, B):
        """
            Returns a Gaussian2D functions and its covariance matrix 
            of the convolution from the two Gaussian2D.
            Args:
                a (numpy.ndarray): mean 2x1 vector of the first Gaussian2D. 
                b (numpy.ndarray): mean 2x1 vector of the second Gaussian2D.
                A (numpy.ndarray): Covariance 2x2 matrix of the first Gaussian2D.
                B (numpy.ndarray): Covariance 2x2 matrix of the second Gaussian2D.
        """
        x_mean = (a+b)[0]
        y_mean = (a+b)[1]
        cov_c = A+B
        denom = (2*np.pi*np.linalg.det(cov_c)**(1/2))
        g2d_c = Gaussian2D(
            amplitude=1/denom,
            x_mean=x_mean,
            y_mean=y_mean,
            cov_matrix=cov_c
        )
        return g2d_c, cov_c

    def func_exp(x, a, b, c):
        return a*expit(-b*x)+c

    def func_exp_inc(x, a, b, c):
        return a*expit(b*x)+c

    def func_lin(x, a, b):
        return a*x+b

    def func_const(x, a):
        return a


class Candidate:
    """Model, true data and likelihoods, p_ratios of one candidate.
    Attributes:
        cc_true_data: True data from df_survey.
        cc_true_data: Model data based on host star fits.
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
        back_true (str): true companion or background object
    """
    def __init__(self, df_survey, index_candidate, host_star, band, catalogue):
        """
        Args:
            df_survey (pandas.DataFrame): Data of the candidates of a single host star.
            index_candidate (int): index integer of the candidate in df_survey.
            host_star (Class Object): Previously initiated class for the host star.
            band (str): Band which the candidate was observed in df_survey (columnname).
            catalogue (str): Name of the catalogue the model is based on: gaia, gaiacalctmass or tmass.
        """
        cc_true_data = df_survey.iloc[index_candidate][[
            'pmra_mean',
            'pmdec_mean',
            'pmra_abs',
            'pmdec_abs',
            'pmra_pmdec_corr',
            'pmra_error',
            'pmdec_error',
            'sep',
            'final_uuid',
            band]].to_dict()

        pm_options = ['pmra', 'pmdec']
        y_options = ['mean', 'stddev']
        cc_model_data = {}
        for pm_value in pm_options:
            for y_option in y_options:
                column = f'{pm_value}_{y_option}_model_coeff_{catalogue}'
                host_star_data = host_star.__getattribute__(column)
                if y_option == 'mean' or len(host_star_data) == 2:
                    cc_model_data[pm_value+'_'+y_option] = host_star_data[0]*cc_true_data[band]+host_star_data[1]
                elif y_option == 'stddev':
                    cc_model_data[pm_value+'_'+y_option] = host_star_data[0]*np.exp(-host_star_data[1]*cc_true_data[band])+host_star_data[2]
        column = f'pmra_pmdec_model_{catalogue}'
        cc_model_data['rho_mean'] = host_star.__getattribute__(column)
        cc_model_data['rho_stddev'] = host_star.__getattribute__(column)
        self.cc_true_data = cc_true_data
        self.cc_model_data = cc_model_data
        self.final_uuid = cc_true_data['final_uuid']

    def calc_liklihoods(self, host_star, sigma_model_min, sigma_cc_min):
        """
        Args:
            host_star (Class object): Previously initiated class for the host star.
            sigma_model_min (float or int): The inflating factor for the model likelihood.
            sigma_cc_min (float or int): The inflating factor for its likelihood.
        """
        g2d_model, cov_model = HelperFunctions.get_g2d_func(
            self.cc_model_data['pmra_mean']-host_star.pmra,
            self.cc_model_data['pmdec_mean']-host_star.pmdec,
            np.sqrt(self.cc_model_data['pmra_stddev']**2+sigma_model_min**2),
            np.sqrt(self.cc_model_data['pmdec_stddev']**2+sigma_model_min**2),
            self.cc_model_data['rho_mean']
        )
        g2d_cc, cov_cc = HelperFunctions.get_g2d_func(
            self.cc_true_data['pmra_mean'],
            self.cc_true_data['pmdec_mean'],
            np.sqrt(self.cc_true_data['pmra_error']**2+sigma_cc_min**2),
            np.sqrt(self.cc_true_data['pmdec_error']**2+sigma_cc_min**2),
            self.cc_true_data['pmra_pmdec_corr']
        )
        g2d_conv, cov_conv = HelperFunctions.convolution2d(
            np.array([
                [self.cc_model_data['pmra_mean']-host_star.pmra],
                [self.cc_model_data['pmdec_mean']-host_star.pmdec]
            ]),
            np.array([[0], [0]]),
            cov_model,
            cov_cc
        )
        g2d_pmuM1, cov_pmuM1 = HelperFunctions.get_g2d_func(
            0,
            0,
            np.sqrt(self.cc_true_data['pmra_error']**2+sigma_cc_min**2),
            np.sqrt(self.cc_true_data['pmdec_error']**2+sigma_cc_min**2),
            rho=0
        )
        self.g2d_model = g2d_model
        self.g2d_conv = g2d_conv
        self.g2d_cc = g2d_cc
        self.g2d_pmuM1 = g2d_pmuM1
        self.cov_model = cov_model
        self.cov_cc = cov_cc
        self.cov_conv = cov_conv
        self.cov_pmuM1 = cov_pmuM1

    def calc_prob_ratio(self):
        """Calculates the odds ratio based on the modelled g2d functions."""
        #  Calculate ratio and statement
        #  Adding the text to top right panel
        p_conv = self.g2d_conv(
            self.cc_true_data['pmra_mean'],
            self.cc_true_data['pmdec_mean']
            )[0]
        pmuM1 = self.g2d_pmuM1(
            self.cc_true_data['pmra_mean'],
            self.cc_true_data['pmdec_mean']
            )
        p_ratio = pmuM1/p_conv
        if p_ratio > 1:
            back_true = 'true companion'
        else:
            back_true = 'background object'
        self.p_b = p_conv
        self.p_tc = pmuM1
        self.p_ratio = np.log(p_ratio)
        self.back_true = back_true


class HostStar:
    """Host star of the candidates. 
    Properties of the host star have the units given in gaiadr3.gaia_source.

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
        object_found (Boolean):
        cone_tmass_cross (pandas.DataFrame): Containing the cone cross matched objects.
        cone_tmass_cross (pandas.DataFrame): containing the cone Gaia objects.
        candidates (pandas.DataFrame): Containing id, p_ratio and p_ratio_catalogue.

        Fitted parameters:
        Syntax:
            variable_{mean or stddev}_{coeff or cov}_{catalogue name}
        Coeff attributes contain the fitting coefficients:
            len(coeff)=3: Fitted with HelperFunctions.func_exp.
            len(coeff)=2: Fitted with HelperFunctions.func_lin.
            len(coeff)=1: Fitted with HelperFunctions.const.
        E.g.:
            parallax_stddev_model_coeff_gaiacalctmass (numpy.array): Model parameters.
            pmra_mean_model_coeff_gaiacalc (numpy.array): Model parameters.
            Or print all the attributes print(dir(class_name)).

    """
    object_found = False

    def __init__(self, target):
        """
        Searches for the given target id in the Simbad database
        for the Gaia source id and returns the data on the star.
        Args:
            target (str): Name of the target.
        """
        customSimbad = Simbad()
        customSimbad.add_votable_fields('ids')
        simbad_object_ids = customSimbad.query_object(target)['IDS'][0].split('|')
        if len(simbad_object_ids) >= 1:
            for identifier in simbad_object_ids:
                if 'Gaia DR3' in identifier:
                    source_id = identifier[8:]
                    catalogue = 'gaiadr3.gaia_source'
                    break
                elif 'Gaia DR2' in identifier:
                    source_id = identifier[8:]
                    catalogue = 'gaiadr2.gaia_source'
                elif 'Gaia DR1' in identifier:
                    source_id = identifier[8:]
                    catalogue = 'gaiadr1.gaia_source'
                else:
                    catalogue = False
                    self.object_found = False
            if catalogue:
                sql_query = f"""SELECT ra, ra_error, dec, dec_error, ref_epoch,
                parallax, parallax_error, pmra, pmdec, pmra_error,
                pmdec_error, pmra_pmdec_corr, parallax_pmra_corr,
                parallax_pmdec_corr, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
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
                except HTTPError as hperr:
                    print('Received HTTPError', hperr)
        else:
            self.object_found = False
            print('Not found in Simbad.')

    def cone_tmasscross_objects(self, cone_radius):
        """ Cone around the target star and cross match with 2MASS.
        Args:
            cone_radius (float): Search cone radius in degree.
        """
        if self.object_found:
            job = Gaia.launch_job_async(f"""SELECT *
            FROM gaiadr3.gaia_source AS gaia
            JOIN gaiaedr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
            JOIN gaiaedr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
            JOIN gaiadr1.tmass_original_valid AS tmass
            ON xjoin.original_psc_source_id = tmass.designation
            WHERE 1 = CONTAINS(
                POINT({self.ra}, {self.dec}),
                CIRCLE(gaia.ra, gaia.dec, {cone_radius}))""")
            try:
                cone_objects = job.get_results().to_pandas()
                self.cone_tmass_cross = cone_objects
            except HTTPError as hperr:
                print('Received HTTPError', hperr)

    def cone_gaia_objects(self, cone_radius):
        """ Cone around the target star and colour transform G-Band to K_S-Band.
        Args:
            cone_radius (float): Search cone radius in degree.
        """
        if self.object_found:
            job = Gaia.launch_job_async(f"""SELECT *
            FROM gaiadr3.gaia_source AS gaia
            WHERE 1 = CONTAINS(
                POINT({self.ra}, {self.dec}),
                CIRCLE(gaia.ra, gaia.dec, {cone_radius}))""")
            try:
                cone_objects = job.get_results().to_pandas()
                cone_objects['phot_bp_rp_mean_mag'] = cone_objects['phot_bp_mean_mag']-cone_objects['phot_rp_mean_mag']
                cone_objects['ks_m_calc'] = cone_objects['phot_g_mean_mag']
                + 0.0981
                - 2.089*cone_objects['phot_bp_rp_mean_mag']
                + 0.1579*(cone_objects['phot_bp_rp_mean_mag'])**2
                self.cone_gaia = cone_objects

            except HTTPError as hperr:
                print('Received HTTPError', hperr)

    def binning_parameters(self, df, x_col_name, y_col_name, binsize, band):
        """ Gaussian 2D fits to each bin.
        Args:
            df (pandas.DataFrame): Cone catalouge data.
            x_col_name (str): pmra, pmdec or parallax.
            y_col_name (str): pmra, pmdec or parallax.
            binsize (float): Number of objects in each single bin.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.
        Returns:
            catalogue_bin_parameters (pandas.DataFrame): Parameters for each bin and each correlation.
        """
        nofbins = int(len(df[band].dropna())/binsize)

        df_dropna = df[[x_col_name, y_col_name, band]].dropna()
        df_sorted = df_dropna[band].sort_values()
        x_stds, y_stds, x_means, y_means, magnitudes, covs, rhos = ([] for i in range(7))
        for samplesize in range(1, nofbins+1):
            bin_width = samplesize*binsize
            m_min = df_sorted.iloc[bin_width-binsize:bin_width].min()
            m_max = df_sorted.iloc[bin_width-binsize:bin_width].max()
            df_r = df_dropna[df_dropna[band].between(m_min, m_max)]
            if len(df_r) > 1:
                x_data, y_data = HelperFunctions.convert_df_to_array(df_r, x_col_name, y_col_name)
                amplitude, x_mean, y_mean, x_stddev, y_stddev, cov, rho = HelperFunctions.get_g2d_parameters(x_data, y_data)
                x_stds.append(x_stddev)
                y_stds.append(y_stddev)
                x_means.append(x_mean)
                y_means.append(y_mean)
                covs.append(cov)
                rhos.append(rho)
                magnitudes.append((m_max+m_min)/2)
        catalogue_bin_parameters = pd.DataFrame({
                f'{x_col_name}_mean':x_means,
                f'{y_col_name}_mean':y_means,
                f'{x_col_name}_stddev':x_stds,
                f'{y_col_name}_stddev':y_stds,
                f'cov_{x_col_name}_{y_col_name}':covs,
                'band':magnitudes,
                f'rho_{x_col_name}_{y_col_name}':rhos,
            })
        return catalogue_bin_parameters


    def concat_binning_parameters(self, df_catalogue, band):
        """ Concat the binning parameters of the combinations of pmra, pmdec, parallax.
        Args:
            df_catalogue (pandas.DataFrame): Cone catalouge data.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.
        Returns:
            df_catalogue_bp (pandas.DataFrame): Binning parameters of different catalogues and 
                variables parameters in a single dataframe.
        """
        catalogue_bp = {}
        for combination in list(itertools.combinations(['pmra', 'pmdec', 'parallax'], r=2)):
            catalogue_bp[f'{combination[0]}_{combination[1]}'] = self.binning_parameters(
                df_catalogue, 
                combination[0],
                combination[1],
                binsize=50, 
                band=band
            )
        df_catalogue_bp = pd.concat([catalogue_bp[key] for key in catalogue_bp.keys()], axis=1)
        df_catalogue_bp = df_catalogue_bp.loc[:,~df_catalogue_bp.columns.duplicated()].copy()
        return df_catalogue_bp


    def pmm_parameters(self, list_of_df_bp, band, candidates_df, include_candidates):
        """ Fit the binning parameters.
        Args:
            list_of_df_bp (list of pandas.DataFrame s): Binned 2D Gaussian Parameters.
            band (str): Bandwidth from the catalogue column
                 e.g. 'ks_m_calc' for Gaia or 'ks_m' for 2MASS.
            candidates_df (pandas.DataFrame): Data on all candidates of this host star.
            include_candidates (Boolean): Including the data of the caniddates in the fitting.
        Attributes:
            For each catalogue and variable there are coeff and cov attributes.
            The syntax:
                variable _ mean or stddev _ coeff or cov _ catalogue name
            The coeff attributes contain the fitting coefficients:
                len(coeff)=3: Fitted with HelperFunctions.func_exp.
                len(coeff)=2: Fitted with HelperFunctions.func_lin.
                len(coeff)=1: Fitted with HelperFunctions.const.
        """
        df_label = ['gaiacalc', 'tmass', 'gaiacalctmass']
        concated_data = pd.concat(list_of_df_bp)
        for idx, df in enumerate([*list_of_df_bp, concated_data]):
            for y_option in ['mean', 'stddev']:
                for pm_value in ['pmra', 'pmdec', 'parallax']:
                    #  Fitting with robust regression
                    lower_clip = np.percentile(df[band], q=10)
                    higher_clip = np.percentile(df[band], q=90)
                    #  Shorten the data 
                    df_clipped = df[df[band].between(lower_clip, higher_clip)]  
                    data_g2m = df_clipped[[band, f'{pm_value}_{y_option}']].dropna()
                    x_data = data_g2m[band].values
                    y_data = data_g2m[f'{pm_value}_{y_option}'].values
                    if include_candidates:
                        np.append(x_data, candidates_df[band].values)
                        np.append(y_data, candidates_df[pm_value+'_abs'].values)
                    if y_option == 'mean' and pm_value in ['pmra', 'pmdec']:
                        fitting_func = HelperFunctions.func_lin
                        boundaries = ([-np.inf, -np.inf], [np.inf, np.inf])
                    elif y_option=='stddev' and pm_value in ['pmra', 'pmdec']:
                        fitting_func = HelperFunctions.func_exp
                        boundaries = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                    elif pm_value=='parallax' and y_option=='mean':
                        fitting_func = HelperFunctions.func_const
                        boundaries = ([-np.inf],[np.inf])
                    elif pm_value=='parallax' and y_option=='stddev':
                        fitting_func = HelperFunctions.func_const
                        boundaries = ([-np.inf],[np.inf])
                    try:
                        popt, pcov = curve_fit(fitting_func, x_data, y_data, bounds=boundaries)
                        attr_name = f'{pm_value}_{y_option}_model_coeff_{df_label[idx]}'
                        setattr(self, attr_name, popt)
                        attr_name = f'{pm_value}_{y_option}_model_cov_{df_label[idx]}'
                        setattr(self, attr_name, pcov)
                    except (RuntimeError, OptimizeWarning):
                        if y_option=='stddev':
                            try:
                                popt, pcov = curve_fit(HelperFunctions.func_lin, x_data, y_data)
                                attr_name = f'{pm_value}_{y_option}_model_coeff_{df_label[idx]}'
                                setattr(self, attr_name, popt)
                                attr_name = f'{pm_value}_{y_option}_model_cov_{df_label[idx]}'
                                setattr(self, attr_name, pcov)
                            except (RuntimeError, OptimizeWarning):
                                print('Fitting error', y_option, pm_value)
            for col in df.columns:
                if 'rho' in col:
                    col1 = re.split('_', col)[1]+'_mean'
                    col2 = re.split('_', col)[2]+'_mean'
                    data_g2m = df[[col1, col2]].dropna()
                    x_data = data_g2m[col1].values
                    y_data = data_g2m[col2].values
                    corr = np.corrcoef(np.array([x_data,y_data]))[0,1]
                    attr_name = f'{col[4:]}_model_{df_label[idx]}'
                    setattr(self, attr_name, corr)
    

    def candidates_table(self, candidates_df, sigma_model_min, sigma_cc_min):
        """
        Returns all the candidates data of this host star for both catalogues.
        Args:
            candidates_df (pandas.DataFrame): Data on all candidates of this host star.
            sigma_model_min (float or int): The inflating factor for the model likelihood.
            sigma_cc_min (float or int): The inflating factor for its likelihood.
        """
        final_uuids, p_ratios, catalogues = ([] for i in range(3)) 
        for index_candidate in range(len(candidates_df)):
            for catalogue in ['tmass', 'gaiacalctmass']:
                #  Create candidate object
                candidate = Candidate(
                    candidates_df, 
                    index_candidate=index_candidate,
                    host_star=self,
                    band='band',
                    catalogue=catalogue
                )
                # Compute liklihoods
                candidate.calc_liklihoods(self, sigma_model_min, sigma_cc_min)
                # Compute odds ratio
                candidate.calc_prob_ratio()
                final_uuids.append(candidate.final_uuid)
                p_ratios.append(candidate.p_ratio)
                catalogues.append(catalogue)
        candidates_p_ratios = pd.DataFrame({
            'final_uuid':final_uuids,
            'p_ratio':p_ratios,
            'p_ratio_catalogue':catalogues
        })
        candidates = candidates_df.merge(candidates_p_ratios, on=['final_uuid'])
        self.candidates = candidates