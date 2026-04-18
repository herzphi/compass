"""
README end-to-end example — run this script to visually inspect COMPASS output.

Uses the 3-epoch synthetic candidate from the README (near HIP82545).
Requires network access to Gaia and Simbad.

Usage:
    pixi run python run_example.py
"""

import numpy as np
import pandas as pd

from compass import model, preset_plots

# ---------------------------------------------------------------------------
# Input data (from README visualization example)
# 3 epochs, nearly static relative position → expected to favour companion
# ---------------------------------------------------------------------------
df_test = pd.DataFrame({
    "Main_ID":        ["HIP82545", "HIP82545", "HIP82545"],
    "final_uuid":     ["id0",       "id0",       "id0"],
    "date":           ["2016-04-24", "2018-04-24", "2021-06-04"],
    "dRA":            [-1390,        -1386,        -1394],
    "dRA_err":        [5,            5,            5],
    "dDEC":           [-990,         -989,         -995],
    "dDEC_err":       [5,            5,            5],
    "dRA_dDEC_corr":  [0,            0,            0],
    "mag0":           [16,           16,           16],
    "mag0_err":       [1,            1,            1],
})

# ---------------------------------------------------------------------------
# Step 1: Ingest observations
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Initialising Survey (queries Simbad + Gaia)...")
print("=" * 60)
survey = model.Survey(df_test, "mag0")

print(f"\nTargets found: {survey.target_names}")
candidates_data = survey.candidates_data["HIP82545"]
print(f"\nPreprocessed candidates table ({len(candidates_data)} row):")
print(candidates_data[["final_uuid", "band", "sep", "pmra_mean", "pmdec_mean",
                        "pmra_error", "pmdec_error"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Step 2: Build field star model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Building field star model (cone search around HIP82545)...")
print("=" * 60)
survey.set_fieldstar_models("ks_m_calc", "ks_m", cone_radius=0.1, binsize=50)

host_star = survey.fieldstar_models["HIP82545"]
print(f"\nHost star astrometry from Gaia DR3:")
print(f"  pmra  = {host_star.pmra:.3f} ± {host_star.pmra_error:.3f} mas/yr")
print(f"  pmdec = {host_star.pmdec:.3f} ± {host_star.pmdec_error:.3f} mas/yr")
print(f"  plx   = {host_star.parallax:.3f} ± {host_star.parallax_error:.3f} mas")

print("\nBackground model coefficients (gaiacalctmass catalogue):")
cat = host_star.background_model_coeffs["gaiacalctmass"]
for param in ["pmra_mean", "pmdec_mean", "pmra_stddev", "pmdec_stddev",
              "parallax_mean", "parallax_stddev"]:
    coeffs = cat[f"{param}_coeff"]
    print(f"  {param:20s}: {np.array2string(coeffs, precision=4)}")

# ---------------------------------------------------------------------------
# Step 3: Evaluate candidates
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Evaluating candidates...")
print("=" * 60)
survey.set_evaluated_fieldstar_models(sigma_cc_min=0, sigma_model_min=0)

candidates = host_star.candidates
print(f"\nResults ({len(candidates)} rows, one per candidate × catalogue):")
cols = ["final_uuid", "r_tcb_catalogue", "r_tcb_2Dnmodel", "r_tcb_pmmodel", "band", "sep"]
print(candidates[cols].to_string(index=False))

# ---------------------------------------------------------------------------
# Step 4: True companions above threshold
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: get_true_companions(threshold=0)")
print("=" * 60)
true_companions = survey.get_true_companions(threshold=0)
if len(true_companions):
    print(f"\n{len(true_companions)} candidate(s) above threshold:")
    print(true_companions[["final_uuid", "r_tcb_catalogue",
                            "r_tcb_2Dnmodel", "r_tcb_pmmodel"]].to_string(index=False))
else:
    print("\nNo candidates above threshold.")

# ---------------------------------------------------------------------------
# Step 5: Covariance matrices
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Covariance matrix inspection (gaiacalctmass, first candidate)")
print("=" * 60)
row = candidates[candidates["r_tcb_catalogue"] == "gaiacalctmass"].iloc[0]
cov_b = np.array(row["cov_background_object"])
cov_tc = np.array(row["cov_true_companion"])
cov_obs = np.array(row["cov_measured_positions"])
print(f"\ncov_background_object  ({cov_b.shape}):\n{np.array2string(cov_b, precision=2)}")
print(f"\ncov_true_companion     ({cov_tc.shape}):\n{np.array2string(cov_tc, precision=2)}")
print(f"\ncov_measured_positions ({cov_obs.shape}):\n{np.array2string(cov_obs, precision=2)}")

print(f"\nmean_background_object:   {np.array2string(np.array(row['mean_background_object']), precision=2)}")
print(f"mean_true_companion:       {np.array2string(np.array(row['mean_true_companion']), precision=2)}")
print(f"mean_measured_positions:   {np.array2string(np.array(row['mean_measured_positions']), precision=2)}")

# ---------------------------------------------------------------------------
# Step 6: Plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Generating plots...")
print("=" * 60)
import matplotlib.pyplot as plt

candidate_obj = host_star.candidates_objects[0]

fig1, _ = plt.subplots()
plt.close(fig1)  # close blank; p_ratio_plot creates its own figure
preset_plots.p_ratio_plot(candidate_obj, "HIP82545", "band")
plt.suptitle("PM-based odds ratio diagnostic", y=1.01)
plt.tight_layout()
print("  → p_ratio_plot displayed")
plt.show(block=False)

fig2, _ = preset_plots.p_ratio_relative_position(
    candidates[candidates["r_tcb_catalogue"] == "gaiacalctmass"],
    "HIP82545",
)
print("  → p_ratio_relative_position displayed")
plt.show(block=False)

print("\nDone. Close plot windows to exit.")
plt.show()
