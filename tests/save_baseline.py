"""
Generate the numerical baseline from the README example.

Run once after confirming the output is correct, then commit the
resulting files in tests/baselines/ as the reference for future
regression tests.

Usage:
    pixi run python tests/save_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compass import model

BASELINES_DIR = Path(__file__).parent / "baselines"
BASELINES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run the full pipeline (same data as test_end_to_end.py)
# ---------------------------------------------------------------------------
df_test = pd.DataFrame({
    "Main_ID":       ["HIP82545", "HIP82545", "HIP82545"],
    "final_uuid":    ["id0",       "id0",       "id0"],
    "date":          ["2016-04-24", "2018-04-24", "2021-06-04"],
    "dRA":           [-1390,        -1386,        -1394],
    "dRA_err":       [5,            5,            5],
    "dDEC":          [-990,         -989,         -995],
    "dDEC_err":      [5,            5,            5],
    "dRA_dDEC_corr": [0,            0,            0],
    "mag0":          [16,           16,           16],
    "mag0_err":      [1,            1,            1],
})

print("Running pipeline...")
survey = model.Survey(df_test, "mag0")
survey.set_fieldstar_models("ks_m_calc", "ks_m", cone_radius=0.1, binsize=50)
survey.set_evaluated_fieldstar_models(sigma_cc_min=0, sigma_model_min=0)

host_star = survey.fieldstar_models["HIP82545"]
candidates = host_star.candidates

# ---------------------------------------------------------------------------
# Collect outputs to save
# ---------------------------------------------------------------------------

# One row per catalogue for the single candidate
row_tmass       = candidates[candidates["r_tcb_catalogue"] == "tmass"].iloc[0]
row_gaiacalc    = candidates[candidates["r_tcb_catalogue"] == "gaiacalctmass"].iloc[0]

# Scalars: odds ratios
scalars = {
    "r_tcb_2Dnmodel_tmass":        float(row_tmass["r_tcb_2Dnmodel"]),
    "r_tcb_pmmodel_tmass":         float(row_tmass["r_tcb_pmmodel"]),
    "r_tcb_2Dnmodel_gaiacalctmass": float(row_gaiacalc["r_tcb_2Dnmodel"]),
    "r_tcb_pmmodel_gaiacalctmass":  float(row_gaiacalc["r_tcb_pmmodel"]),
    # Host star astrometry
    "host_pmra":            float(host_star.pmra),
    "host_pmdec":           float(host_star.pmdec),
    "host_parallax":        float(host_star.parallax),
    # Background model coefficients (gaiacalctmass)
    "pmra_mean_coeff":   list(host_star.background_model_coeffs["gaiacalctmass"]["pmra_mean_coeff"].tolist()),
    "pmdec_mean_coeff":  list(host_star.background_model_coeffs["gaiacalctmass"]["pmdec_mean_coeff"].tolist()),
    "pmra_stddev_coeff": list(host_star.background_model_coeffs["gaiacalctmass"]["pmra_stddev_coeff"].tolist()),
    "pmdec_stddev_coeff":list(host_star.background_model_coeffs["gaiacalctmass"]["pmdec_stddev_coeff"].tolist()),
    "parallax_mean":     float(host_star.background_model_coeffs["gaiacalctmass"]["parallax_mean_coeff"][0]),
    "parallax_stddev":   float(host_star.background_model_coeffs["gaiacalctmass"]["parallax_stddev_coeff"][0]),
}

# Arrays: means and covariance matrices (gaiacalctmass row)
arrays = {
    "mean_measured_positions":  np.array(row_gaiacalc["mean_measured_positions"]),
    "mean_true_companion":      np.array(row_gaiacalc["mean_true_companion"]),
    "mean_background_object":   np.array(row_gaiacalc["mean_background_object"]),
    "cov_measured_positions":   np.array(row_gaiacalc["cov_measured_positions"]),
    "cov_true_companion":       np.array(row_gaiacalc["cov_true_companion"]),
    "cov_background_object":    np.array(row_gaiacalc["cov_background_object"]),
}

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
scalars_path = BASELINES_DIR / "readme_example_scalars.json"
arrays_path  = BASELINES_DIR / "readme_example_arrays.npz"

with open(scalars_path, "w") as f:
    json.dump(scalars, f, indent=2)

np.savez(arrays_path, **arrays)

print(f"\nBaseline saved:")
print(f"  {scalars_path}")
print(f"  {arrays_path}")
print("\nKey values:")
print(f"  r_tcb_2Dnmodel (gaiacalctmass) = {scalars['r_tcb_2Dnmodel_gaiacalctmass']:.4f}")
print(f"  r_tcb_pmmodel  (gaiacalctmass) = {scalars['r_tcb_pmmodel_gaiacalctmass']:.4f}")
print(f"  r_tcb_2Dnmodel (tmass)         = {scalars['r_tcb_2Dnmodel_tmass']:.4f}")
print(f"  r_tcb_pmmodel  (tmass)         = {scalars['r_tcb_pmmodel_tmass']:.4f}")
