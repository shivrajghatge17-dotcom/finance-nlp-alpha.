import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimizer (Markowitz Model)")

# --- Sidebar Inputs ---
st.sidebar.header("User Inputs")
tickers = st.sidebar.text_input(
    "Enter stock tickers (comma-separated)",
    "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, ASML, ADBE, AVGO"
).split(",")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
num_portfolios = st.sidebar.slider("Number of Portfolios to Simulate", 1000, 20000, 5000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.04)

# --- Fetch Data ---
data = yf.download([t.strip() for t in tickers], start=start_date, end=end_date)["Close"]
returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# --- Monte Carlo Simulation ---
port_returns, port_volatility, sharpe_ratios, port_weights = [], [], [], []
for _ in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    port_weights.append(weights)
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    port_returns.append(ret)
    port_volatility.append(vol)
    sharpe_ratios.append(sharpe)

results = pd.DataFrame({
    "Return": port_returns,
    "Volatility": port_volatility,
    "Sharpe Ratio": sharpe_ratios
})

# --- Find Optimal Portfolios ---
max_sharpe = results.loc[results["Sharpe Ratio"].idxmax()]
min_vol = results.loc[results["Volatility"].idxmin()]

# --- Plot Efficient Frontier ---
fig = px.scatter(
    results,
    x="Volatility",
    y="Return",
    color="Sharpe Ratio",
    color_continuous_scale="Viridis",
    title="Efficient Frontier",
    labels={"Volatility": "Risk", "Return": "Expected Return"}
)
fig.add_scatter(x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]],
                mode="markers+text", text=["Max Sharpe"], textposition="top center",
                marker=dict(color="red", size=12))
fig.add_scatter(x=[min_vol["Volatility"]], y=[min_vol["Return"]],
                mode="markers+text", text=["Min Volatility"], textposition="top center",
                marker=dict(color="blue", size=12))
st.plotly_chart(fig, use_container_width=True)

# --- Display Optimal Weights ---
weights_df = pd.DataFrame(port_weights[results["Sharpe Ratio"].idxmax()],
                          index=[t.strip() for t in tickers],
                          columns=["Weight"])
st.subheader("Optimal Portfolio Weights (Max Sharpe)")
st.dataframe(weights_df.style.format("{:.2%}"))

# --- Summary Metrics ---
st.subheader("Performance Summary")
st.write(f"**Max Sharpe Portfolio Return:** {max_sharpe['Return']:.2%}")
st.write(f"**Max Sharpe Portfolio Volatility:** {max_sharpe['Volatility']:.2%}")
st.write(f"**Sharpe Ratio:** {max_sharpe['Sharpe Ratio']:.2f}")
