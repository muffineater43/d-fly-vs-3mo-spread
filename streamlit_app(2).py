import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
import warnings

# suppress irrelevant warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style")

st.set_page_config(layout="wide")
st.title("d-fly vs 3mo spread")

# Upload two CSVs
fly_file = st.sidebar.file_uploader("Upload double-butterfly CSV", type="csv")
leg_file = st.sidebar.file_uploader("Upload 3 month spread CSV",   type="csv")
if not fly_file or not leg_file:
    st.sidebar.info("Please upload both a d-fly and an spread CSV.")
    st.stop()

# Load and align on common dates
df_fly = pd.read_csv(fly_file, parse_dates=["Timestamp (UTC)"])
df_leg = pd.read_csv(leg_file, parse_dates=["Timestamp (UTC)"])
common = set(df_fly["Timestamp (UTC)"]) & set(df_leg["Timestamp (UTC)"])
fly_common = df_fly[df_fly["Timestamp (UTC)"].isin(common)].sort_values("Timestamp (UTC)")
leg_common = df_leg[df_leg["Timestamp (UTC)"].isin(common)].sort_values("Timestamp (UTC)")

# build aligned DataFrame
df = pd.DataFrame({
    "leg": leg_common.set_index("Timestamp (UTC)")["Close"],
    "fly": fly_common.set_index("Timestamp (UTC)")["Close"],
}).dropna()
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Rolling 3-month regression
betas, alphas = [], []
min_periods = 50
for t in df.index:
    window_df = df.loc[df.index >= (t - pd.DateOffset(months=3))]
    if len(window_df) < min_periods:
        betas.append(np.nan)
        alphas.append(np.nan)
    else:
        m, b0 = np.polyfit(window_df["leg"], window_df["fly"], 1)
        betas.append(m)
        alphas.append(b0)

# attach regression results
df["beta"]      = betas
df["intercept"] = alphas
df["predicted"] = df["beta"] * df["leg"] + df["intercept"]
df["residual"]  = df["fly"] - df["predicted"]

# Compute and display metrics
mu      = df["residual"].mean()
sigma   = df["residual"].std(ddof=1)
latest  = df["residual"].iloc[-1]
z_score = (latest - mu) / sigma

st.subheader("Latest Residual Z-Score")
st.write(f"Used a true 3-month slice ending at {df.index[-1].date()}")
st.metric("Z-score", f"{z_score:.2f}")

residuals = df["residual"].dropna()
mu2       = residuals.mean()
sigma2    = residuals.std(ddof=1)
skw       = skew(residuals)
kurt_p    = kurtosis(residuals, fisher=False)

st.subheader("Residual Summary Statistics")
st.write(f"Mean:     {mu2:.4f}")
st.write(f"Std Dev:  {sigma2:.4f}")
st.write(f"Skewness: {skw:.4f}")
st.write(f"Kurtosis: {kurt_p:.4f}")

# Optional histogram with fitted normal curve
fig, ax = plt.subplots()
ax.hist(residuals, bins=30, density=True, alpha=0.6)
x = np.linspace(mu2 - 4*sigma2, mu2 + 4*sigma2, 200)
ax.plot(x, norm.pdf(x, mu2, sigma2), linewidth=2)
ax.set_title("Residuals Histogram with Fitted Normal Curve")
st.pyplot(fig, use_container_width=True)

# Save feature: record butterfly name and current metrics
if "history" not in st.session_state:
    st.session_state.history = []

def save_metrics():
    st.session_state.history.append({
        "butterfly": fly_file.name,
        "mean":      mu2,
        "std":       sigma2,
        "skew":      skw,
        "kurtosis":  kurt_p,
        "z_score":   z_score
    })

st.button("Save Metrics", on_click=save_metrics)

# Show and download history
if st.session_state.history:
    st.subheader("Saved Metrics History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Metrics CSV", csv, "metrics_history.csv", "text/csv")
