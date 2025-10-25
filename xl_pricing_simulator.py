# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 21:22:26 2025

@author: Makarand Kulkarni
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import io

st.set_page_config(page_title="XL Reinsurance Pricing Simulation", layout="wide")

# --- Sidebar help ---
with st.sidebar:
    st.header("About inputs")
    st.markdown(
        """
        **File uploader**: CSV with one column of yearly payouts (1D vector). Header optional.

        **Loss Ratio (LR)**: Enter as percent (e.g. 30 for 30%). Must be > 0.

        **stdev_loading**: Scalar applied to standard deviations when computing premium burn rate and layer rates.

        **Sum insured**: Reference sum-insured; used to cap upper bound if needed.

        **XL layers**: Comma-separated list of ranges like `150-175,175-200`. Use hyphen to separate lower-upper.

        **Distribution**: Choose how to simulate payouts (Normal, Lognormal, Gamma).

        **Number of simulations**: How many simulated payout draws to generate.
        """
    )

st.title("XL Reinsurance Pricing Simulation")

# --- Input Parameters ---
st.header("Input Parameters")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload CSV with yearly payouts (one column)", type=["csv"])
    raw_text = st.text_area("Or paste payouts as comma/newline separated numbers (optional)", height=80)
    xl_layers_str = st.text_input("XL layers (comma-separated, e.g. 150-175,175-200,210-250)", "150-175,175-200")
    sum_insured = st.number_input("Sum insured", value=100000.0, format="%.2f")

with col2:
    lr_pct = st.number_input("Loss Ratio (LR %)", min_value=0.001, value=85.0, format="%.3f")
    stdev_loading = st.number_input("stdev_loading (scalar)", value=0.2, format="%.3f")
    dist_type = st.selectbox("Distribution", ["Normal", "Lognormal", "Gamma"]) 
    n_sim = st.number_input("Number of simulations", min_value=100, value=10000, step=100)

st.markdown("---")

# --- Read payout data ---
@st.cache_data
def load_payouts(file, pasted_text):
    if file is not None:
        try:
            df = pd.read_csv(file, header=None)
            # If there is more than 1 column, try to take first numeric column
            if df.shape[1] > 1:
                # pick first numeric column
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        series = df[c].dropna().astype(float)
                        if not series.empty:
                            return series
                # fallback: flatten all values
                vals = pd.to_numeric(df.stack(), errors='coerce').dropna()
                return vals.astype(float)
            else:
                series = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(float)
                return series
        except Exception as e:
            st.warning(f"Failed to read uploaded CSV: {e}")
    # If no file, check pasted text
    if pasted_text and pasted_text.strip():
        # split by comma or newline
        parts = [p.strip() for p in pasted_text.replace('\r','\n').replace(';',',').replace('\n',',').split(',')]
        parts = [p for p in parts if p != '']
        try:
            vals = pd.to_numeric(parts, errors='coerce')
            vals = pd.Series(vals).dropna().astype(float)
            if not vals.empty:
                return vals
        except Exception:
            pass
    return None

payout_series = load_payouts(uploaded, raw_text)

if payout_series is None or payout_series.empty:
    st.error("No valid payout series found. Please upload a CSV or paste numbers. Showing example data instead.")
    # provide small example
    payout_series = pd.Series([100, 120, 90, 200, 150, 80, 300, 50, 400, 250], dtype=float)

#st.write("Sample of payout series:")
#st.dataframe(payout_series.head(10).to_frame(name='payout'))

# --- Simulation Settings ---
st.header("Simulation Settings")
st.write(f"Distribution: **{dist_type}**, Simulations: **{n_sim:,}**, stdev_loading: **{stdev_loading}**, LR: **{lr_pct}%**")

# Derive mean and std
data_mean = payout_series.mean()
data_std = payout_series.std(ddof=1)

# Generate simulations
@st.cache_data
def simulate_payouts(mean, std, dist, n):
    rng = np.random.default_rng()
    if dist == 'Normal':
        sims = rng.normal(loc=mean, scale=std, size=int(n))
    elif dist == 'Lognormal':
        # For lognormal, estimate mu and sigma from positive data
        # if mean/std are 0 or negative values present, fall back to a shifted lognormal
        if std <= 0 or mean <= 0:
            # fallback to small positive values
            mu = np.log(max(1e-6, mean + 1))
            sigma = max(1e-6, std / max(mean, 1e-6))
        else:
            # approximate parameters of underlying normal: mu_ln and sigma_ln
            variance = std ** 2
            mu_ln = np.log(max(1e-12, mean ** 2 / np.sqrt(variance + mean ** 2)))
            sigma_ln = np.sqrt(np.log(1 + variance / (mean ** 2)))
            mu, sigma = mu_ln, sigma_ln
        sims = rng.lognormal(mean=mu, sigma=sigma, size=int(n))
    elif dist == 'Gamma':
        # Fit a gamma to positive data if possible
        # Here we approximate shape/scale from mean and std: shape = (mean/std)^2, scale = std^2/mean
        if std <= 0 or mean <= 0:
            shape = 1.0
            scale = max(1.0, mean)
        else:
            shape = (mean / std) ** 2
            scale = (std ** 2) / mean
        sims = rng.gamma(shape, scale, size=int(n))
    else:
        sims = rng.normal(loc=mean, scale=std, size=int(n))
    # ensure non-negative payouts
    sims = np.where(np.isfinite(sims), sims, 0.0)
    sims[sims < 0] = 0.0
    return sims

sim_payout = simulate_payouts(data_mean, data_std, dist_type, n_sim)

# Compute premium_burnrate_amount
premium_burnrate_amount = np.mean(sim_payout) + stdev_loading * np.std(sim_payout, ddof=1)

# Parse XL layers robustly
def parse_layers(layers_str):
    tokens = [t.strip() for t in layers_str.split(',') if t.strip()]
    parsed = []
    for tok in tokens:
        if '-' in tok:
            a, b = tok.split('-', 1)
            try:
                low = float(a)
                up = float(b)
                if up < low:
                    low, up = up, low
                parsed.append((low, up))
            except Exception:
                continue
        else:
            # single number; treat as a layer from that number to that+1
            try:
                v = float(tok)
                parsed.append((v, v))
            except Exception:
                continue
    return parsed

parsed_layers = parse_layers(xl_layers_str)
if not parsed_layers:
    st.error("No valid XL layers parsed. Please provide layers like 150-175,175-200")
    st.stop()

# Ensure continuous bands conceptually by building boundary arrays (for internal checking)
# Xl_lower = [0, lower1, lower2, ...]
# Xl_upper = [upper1, upper2, ..., 100000]
#lower_bounds = [0.0] + [p[0] for p in parsed_layers]
#upper_bounds = [p[1] for p in parsed_layers] + [max(sum_insured, parsed_layers[-1][1], 100000.0)]

# --- Ensure continuous coverage of XL layers ---
sorted_layers = sorted(parsed_layers, key=lambda x: x[0])
Xl_lower = [0.0]
Xl_upper = []

last_upper = 0.0
for low, up in sorted_layers:
    if low > last_upper:
        # Fill missing band
        Xl_lower.append(last_upper)
        Xl_upper.append(low)
    Xl_lower.append(low)
    Xl_upper.append(up)
    last_upper = up
    
# Add final open-ended layer up to high limit
max_limit = max(sorted_layers[-1][1], 100000.0)
if Xl_upper[-1] < max_limit:
    Xl_lower.append(Xl_upper[-1])
    Xl_upper.append(max_limit)

# Remove potential duplicates at start
if len(Xl_lower) > 1 and Xl_lower[1] == Xl_lower[0]:
    Xl_lower = Xl_lower[1:]

Xl_lower=np.array(Xl_lower)
Xl_upper=np.array(Xl_upper)    

#st.text(f"parsed_layers={parsed_layers}")
#st.text(f"Xl_lower={Xl_lower}")
#st.text(f"Xl_upper={Xl_upper}")

# Compute scaling factor: premium_burnrate_amount / LR
lr_decimal = lr_pct / 100.0
if lr_decimal <= 0:
    st.error("LR must be > 0%")
    st.stop()
scale = premium_burnrate_amount / lr_decimal
a1=np.mean(sim_payout)
a2=np.std(sim_payout, ddof=1)
#st.text(f"Mean={a1}, Std={a2}, Premium={premium_burnrate_amount}")
a=type(sim_payout)
#st.text(f"typeof simpayout={a}")
# For each original layer compute deductible and cover
deductibles = []
covers = []
for (low, up) in parsed_layers:
    ded = scale * low/100
    cov = scale * up/100 - ded
    # ensure non-negative
    ded = max(0.0, ded)
    cov = max(0.0, cov)
    deductibles.append(ded)
    covers.append(cov)

deductibles_all=Xl_lower* scale /100
covers_all=Xl_upper*scale/100 - deductibles_all
#layer_po_all=[np.mean(p) for p in deductibles_all]
layer_po_avg_all=[]
layer_po_std_all=[]
for i in range(len(deductibles_all)):
    tmpa=np.maximum(sim_payout-deductibles_all[i],0)
    tmpa=np.minimum(tmpa,covers_all[i])
    layer_po_avg_all.append(np.average(tmpa))
    layer_po_std_all.append(np.std(np.minimum(np.maximum(sim_payout-deductibles_all[i],0),
                                       covers_all[i])))

layer_rate_all= np.array( layer_po_avg_all) + stdev_loading * np.array(layer_po_std_all)

# Compute layer payouts matrix (n_sim x n_layers)
sim_payout = np.array(sim_payout)
n_layers = len(parsed_layers)
layer_payouts = np.zeros((sim_payout.shape[0], n_layers), dtype=float)
for j in range(n_layers):
    ded = deductibles[j]
    cap = covers[j]
    # payout per sim: max(min(sim - ded, cap), 0)
    pay = sim_payout - ded
    pay = np.minimum(pay, cap)
    pay = np.maximum(pay, 0.0)
    layer_payouts[:, j] = pay
    # Rate on line (%)
    

# Compute stats
average_layer_po = layer_payouts.mean(axis=0)
# stdev of the vector of average_layer_po
#avg_po_std = np.std(average_layer_po, ddof=1)
avg_po_std = np.std(layer_payouts, axis=0)
#st.text(f"average_layer_po={average_layer_po}")
#st.text(f"avg_po_std={avg_po_std}")

layer_rate = average_layer_po + avg_po_std * stdev_loading

ROL = [(rate / cov) * 100 for rate, cov in zip(layer_rate, covers)]

# guard against negative or zero sum
#total_layer_rate = layer_rate.sum()
total_layer_rate = layer_rate_all.sum()
#st.text(f"total_layer_rate={total_layer_rate}")
if total_layer_rate <= 0:
    xl_rate_pct = np.zeros_like(layer_rate)
else:
    xl_rate_pct = layer_rate / total_layer_rate * 100.0
#Relative_share
# Prepare results table
results_df = pd.DataFrame({
    'Layer': [f"{int(l)}-{int(u)}" if l.is_integer() and u.is_integer() else f"{l:.2f}-{u:.2f}" for l, u in parsed_layers],
    'Deductible': deductibles,
    'Cover': covers,
    'Average Layer PO': average_layer_po,
    'Layer Rate': layer_rate,
    'Relative_share (%)':xl_rate_pct,
    'ROL (%)': ROL
})

# Format numeric outputs to 2 decimal places for display
display_df = results_df.copy()
for col in ['Deductible', 'Cover', 'Average Layer PO', 'Layer Rate', 'Relative_share (%)',
            'ROL (%)']:
    display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}")

# --- Results Table ---
st.header("Results Table")
st.dataframe(display_df.style.set_table_styles([]), use_container_width=True)

# --- Charts ---
st.header("Charts")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Histogram of simulated payouts")
    hist_fig = px.histogram(sim_payout, nbins=80, title="Simulated payouts distribution")
    hist_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(hist_fig, use_container_width=True)

with chart_col2:
    st.subheader("ROL (%)")
    bar_fig = px.bar(results_df, x='Layer', y='ROL (%)', title='ROL (%) by layer')
    bar_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(bar_fig, use_container_width=True)

# --- Download CSV ---
st.header("Download Results")
csv_buf = io.StringIO()
results_df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode('utf-8')

st.download_button("Download layer-wise results as CSV", data=csv_bytes, file_name="xl_layer_results.csv", mime='text/csv')

# Also allow download of full simulation payouts (optional)
sim_df = pd.DataFrame(sim_payout, columns=['sim_payout'])
sim_buf = io.StringIO()
sim_df.to_csv(sim_buf, index=False)
st.download_button("Download simulated payouts (first 10000) as CSV", data=sim_buf.getvalue().encode('utf-8'), file_name='sim_payouts.csv', mime='text/csv')

# --- Footer / summary ---
st.markdown("---")
st.subheader("Summary values")
st.write(f"Premium burn-rate amount: **{premium_burnrate_amount:,.2f}**")
st.write(f"Scale = premium_burnrate_amount / LR = **{scale:,.2f}**")

st.caption("Note: numeric outputs shown rounded to 2 decimals. App attempts to handle common input mistakes; please check inputs if results look unexpected.")
