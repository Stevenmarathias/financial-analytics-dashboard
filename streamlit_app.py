"""
BA870/AC820 — Financial Analytics Team Project
Streamlit Dashboard: Stock Performance Predictor

Team: Arshdeep Singh Oberoi, Mokhinur Talibzhanova, Steven Marathias

This app loads pre-trained models (trained on WRDS Compustat data in Google Colab)
and uses live Yahoo Finance data to predict whether a company's ROA will
improve or decline in the next fiscal year.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import os

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Performance Predictor",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────
# Load pre-trained artifacts (cached so it only runs once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    path = os.path.join(os.path.dirname(__file__), 'deployment_artifacts.pkl')
    return joblib.load(path)

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Could not load deployment_artifacts.pkl: {e}")
    st.stop()

clf = artifacts['best_classifier']
reg = artifacts['best_regressor']
clip_bounds = artifacts['clip_bounds']
feature_columns = artifacts['feature_columns']
industry_stats_by_sic = artifacts['industry_stats_by_sic']
overall_stats = artifacts['overall_stats']
feature_medians = artifacts['feature_medians']
clf_name = artifacts['best_classifier_name']
reg_name = artifacts['best_regressor_name']
predictions_by_year = pd.DataFrame(artifacts['predictions_by_year'])


# ──────────────────────────────────────────────
# Helper: get price data from yfinance
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_price_data(ticker):
    df = yf.download(ticker, start='2010-01-01', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        return df
    df['daily_return'] = df['Close'].pct_change()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['volatility'] = df['daily_return'].rolling(20).std()
    return df.dropna()


# ──────────────────────────────────────────────
# Helper: predict performance for a ticker
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def predict_performance(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        bs = tk.balance_sheet
        fs = tk.financials
    except Exception:
        return None

    def from_info_or_bs(info_key, bs_row):
        val = info.get(info_key, None)
        if val is not None:
            return float(val)
        if bs is not None and not bs.empty and bs_row in bs.index:
            return float(bs.loc[bs_row].dropna().iloc[0])
        return None

    def from_info_or_fs(info_key, fs_row):
        val = info.get(info_key, None)
        if val is not None:
            return float(val)
        if fs is not None and not fs.empty and fs_row in fs.index:
            return float(fs.loc[fs_row].dropna().iloc[0])
        return None

    total_assets  = from_info_or_bs('totalAssets', 'Total Assets')
    total_debt    = from_info_or_bs('totalDebt', 'Total Debt')
    net_income    = from_info_or_fs('netIncomeToCommon', 'Net Income')
    total_revenue = from_info_or_fs('totalRevenue', 'Total Revenue')
    market_cap    = info.get('marketCap', None)
    equity        = from_info_or_bs('totalStockholderEquity', 'Stockholders Equity')
    trailing_pe   = info.get('trailingPE', None)
    sector        = info.get('sector', 'Unknown')
    company_name  = info.get('shortName', ticker)
    sic_code      = info.get('sic', None)

    def to_millions(val):
        return val / 1_000_000 if val is not None else None

    at_m   = to_millions(total_assets)
    debt_m = to_millions(total_debt)
    ni_m   = to_millions(net_income)
    rev_m  = to_millions(total_revenue)
    eq_m   = to_millions(equity)
    mcap_m = to_millions(market_cap)

    ROA        = ni_m / at_m if (ni_m is not None and at_m and at_m > 0) else np.nan
    Debt_ratio = debt_m / at_m if (debt_m is not None and at_m and at_m > 0) else np.nan
    PS_ratio   = mcap_m / rev_m if (mcap_m and rev_m and rev_m > 0) else np.nan
    log_at     = np.log(at_m) if (at_m and at_m > 0) else np.nan
    PE_ratio   = float(trailing_pe) if trailing_pe else np.nan
    ROE        = ni_m / eq_m if (ni_m is not None and eq_m and eq_m != 0) else np.nan

    # Compute earnings_growth from actual prior-year data
    eg_val = np.nan
    eg_source = 'median'
    try:
        if fs is not None and not fs.empty and 'Net Income' in fs.index:
            ni_series = fs.loc['Net Income'].dropna().sort_index()
            if len(ni_series) >= 2:
                ni_prev = float(ni_series.iloc[-2])
                ni_curr = float(ni_series.iloc[-1])
                if abs(ni_prev) > 0:
                    eg_val = (ni_curr - ni_prev) / abs(ni_prev)
                    eg_source = 'actual'
    except Exception:
        pass
    if np.isnan(eg_val):
        eg_val = feature_medians['earnings_growth']

    # Compute ROA_lagged_1_year from actual prior-year data
    roa_lag_val = np.nan
    roa_lag_source = 'median'
    try:
        if (fs is not None and not fs.empty and 'Net Income' in fs.index and
            bs is not None and not bs.empty and 'Total Assets' in bs.index):
            ni_series = fs.loc['Net Income'].dropna().sort_index()
            at_series = bs.loc['Total Assets'].dropna().sort_index()
            if len(ni_series) >= 2 and len(at_series) >= 2:
                ni_prev = float(ni_series.iloc[-2]) / 1_000_000
                at_prev = float(at_series.iloc[-2]) / 1_000_000
                if at_prev > 0:
                    roa_lag_val = ni_prev / at_prev
                    roa_lag_source = 'actual'
    except Exception:
        pass
    if np.isnan(roa_lag_val):
        roa_lag_val = feature_medians['ROA_lagged_1_year']

    feature_dict = {
        'Debt_ratio':        Debt_ratio,
        'PS_ratio':          PS_ratio,
        'log_at':            log_at,
        'ROA_lagged_1_year': roa_lag_val,
        'PE_ratio':          PE_ratio,
        'ROE':               ROE,
        'earnings_growth':   eg_val,
    }

    feature_sources = {
        'Debt_ratio':        'actual' if not np.isnan(Debt_ratio) else 'N/A',
        'PS_ratio':          'actual' if not np.isnan(PS_ratio) else 'N/A',
        'log_at':            'actual' if not np.isnan(log_at) else 'N/A',
        'ROA_lagged_1_year': roa_lag_source,
        'PE_ratio':          'actual' if not np.isnan(PE_ratio) else 'N/A',
        'ROE':               'actual' if not np.isnan(ROE) else 'N/A',
        'earnings_growth':   eg_source,
    }

    feature_clipped = {}
    for feat, (lo, hi) in clip_bounds.items():
        v = feature_dict.get(feat, np.nan)
        if v is not None and not np.isnan(v):
            clipped_v = np.clip(v, lo, hi)
            feature_clipped[feat] = clipped_v != v
            feature_dict[feat] = clipped_v
        else:
            feature_clipped[feat] = False

    X_live = pd.DataFrame([feature_dict])

    prob       = clf.predict_proba(X_live)[0]
    pred_class = clf.predict(X_live)[0]
    pred_roa   = reg.predict(X_live)[0]

    return {
        'ticker':          ticker.upper(),
        'company_name':    company_name,
        'sector':          sector,
        'sic_code':        sic_code,
        'features':        feature_dict,
        'feature_sources': feature_sources,
        'feature_clipped': feature_clipped,
        'ROA_current':     ROA,
        'prediction':      'IMPROVE' if pred_class == 1 else 'DECLINE',
        'confidence':      float(max(prob)),
        'prob_improve':    float(prob[1]),
        'pred_roa_next':   float(pred_roa),
    }


# ──────────────────────────────────────────────
# Helper: get industry stats from pre-computed data
# ──────────────────────────────────────────────
def get_industry_stats(sic_code=None):
    industry_label = 'All Companies'
    stats = overall_stats

    if sic_code is not None:
        try:
            sic_2d = int(str(sic_code)[:2])
            if sic_2d in industry_stats_by_sic:
                stats = industry_stats_by_sic[sic_2d]
                industry_label = f'SIC {sic_2d}xx Industry'
        except (ValueError, TypeError):
            pass

    industry_means = {
        'ROA':        stats['ROA'],
        'Debt_ratio': stats['Debt_ratio'],
        'PS_ratio':   stats['PS_ratio'],
        'PE_ratio':   stats['PE_ratio'],
        'ROE':        stats['ROE'],
    }

    improve_pct = stats['improve_pct']
    decline_pct = 100 - improve_pct

    return industry_means, improve_pct, decline_pct, industry_label


# ──────────────────────────────────────────────
# App UI
# ──────────────────────────────────────────────
st.title("📊 Financial Performance Predictor")
st.markdown(
    "Enter any US stock ticker to predict whether the company's **Return on Assets (ROA)** "
    "will improve or decline next year, based on models trained on "
    "**WRDS Compustat** data (2010–2024)."
)
st.markdown("---")

# Ticker input
col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10,
                           placeholder="e.g. NVDA, TSLA, META")
with col_btn:
    st.write("")  # spacing
    analyze = st.button("🔍 Analyze", use_container_width=True)

if analyze and ticker.strip():
    ticker = ticker.strip().upper()

    with st.spinner(f"Fetching data for {ticker}..."):
        pred = predict_performance(ticker)
        price_df = get_price_data(ticker)

    if pred is None:
        st.error(f"Could not fetch data for **{ticker}**. Check the symbol and try again.")
        st.stop()

    if price_df.empty:
        st.error(f"No price history found for **{ticker}**.")
        st.stop()

    industry_means, improve_pct, decline_pct, industry_label = get_industry_stats(
        sic_code=pred.get('sic_code')
    )

    # ── Prediction Card ──
    roa_display = f"{pred['ROA_current']:.4f}" if not np.isnan(pred['ROA_current']) else "N/A"

    if pred['prediction'] == 'IMPROVE':
        st.success(
            f"### {pred['company_name']} ({pred['ticker']})\n"
            f"**Sector:** {pred['sector']}  \n"
            f"**Current ROA:** {roa_display}  \n\n"
            f"**Predicted next-year performance: IMPROVE**  \n"
            f"Confidence: **{pred['confidence']:.1%}** &nbsp;|&nbsp; "
            f"P(Improve): **{pred['prob_improve']:.1%}**  \n"
            f"Predicted next-year ROA: **{pred['pred_roa_next']:.4f}**"
        )
    else:
        st.error(
            f"### {pred['company_name']} ({pred['ticker']})\n"
            f"**Sector:** {pred['sector']}  \n"
            f"**Current ROA:** {roa_display}  \n\n"
            f"**Predicted next-year performance: DECLINE**  \n"
            f"Confidence: **{pred['confidence']:.1%}** &nbsp;|&nbsp; "
            f"P(Improve): **{pred['prob_improve']:.1%}**  \n"
            f"Predicted next-year ROA: **{pred['pred_roa_next']:.4f}**"
        )

    st.caption(f"Based on WRDS Compustat data ({industry_label}) & {clf_name} / {reg_name}")

    # ── Feature Details Table ──
    with st.expander("Show model input features"):
        feat_data = []
        for feat in feature_columns:
            val = pred['features'][feat]
            source = pred['feature_sources'][feat]
            clipped = pred['feature_clipped'].get(feat, False)
            feat_data.append({
                'Feature': feat,
                'Value': f"{val:.4f}" if not np.isnan(val) else "N/A",
                'Source': source,
                'Clipped': '✂ yes' if clipped else '',
            })
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Chart A: Historical Price ──
    st.subheader(f"{ticker} — Historical Close Price")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=price_df.index, y=price_df['Close'],
        mode='lines', name='Close Price',
        line=dict(color='#1976d2', width=1.5),
    ))
    fig_line.add_trace(go.Scatter(
        x=price_df.index, y=price_df['ma_20'],
        mode='lines', name='20-Day MA',
        line=dict(color='#ff9800', width=1, dash='dash'),
    ))
    fig_line.update_layout(
        template='plotly_white', height=420,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(count=5, label='5Y', step='year', stepmode='backward'),
                dict(label='All', step='all'),
            ]),
            rangeslider=dict(visible=True),
        ),
        yaxis_title='Price ($)',
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Chart B: Volume vs Return ──
    st.subheader(f"{ticker} — Volume vs Daily Return")
    scatter_df = price_df[['Volume', 'daily_return']].dropna().copy()
    if len(scatter_df) > 2000:
        scatter_df = scatter_df.sample(2000, random_state=42)
    fig_scatter = px.scatter(
        scatter_df, x='Volume', y='daily_return',
        opacity=0.4,
        labels={'Volume': 'Trading Volume', 'daily_return': 'Daily Return'},
        template='plotly_white',
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Chart C: Ratios vs Industry (split panels) ──
    st.subheader(f"{ticker} Ratios vs {industry_label} Median")
    small_ratios = ['ROA', 'Debt_ratio', 'ROE']
    large_ratios = ['PS_ratio', 'PE_ratio']

    company_vals_small = [
        pred['ROA_current'],
        pred['features']['Debt_ratio'],
        pred['features']['ROE'],
    ]
    industry_vals_small = [industry_means[r] for r in small_ratios]
    company_vals_large = [
        pred['features']['PS_ratio'],
        pred['features']['PE_ratio'],
    ]
    industry_vals_large = [industry_means[r] for r in large_ratios]

    fig_bar = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Profitability & Leverage', 'Valuation Multiples'],
        horizontal_spacing=0.15,
    )
    fig_bar.add_trace(go.Bar(
        name=ticker, x=small_ratios, y=company_vals_small,
        marker_color='#1976d2', showlegend=True,
    ), row=1, col=1)
    fig_bar.add_trace(go.Bar(
        name=industry_label, x=small_ratios, y=industry_vals_small,
        marker_color='#9e9e9e', showlegend=True,
    ), row=1, col=1)
    fig_bar.add_trace(go.Bar(
        name=ticker, x=large_ratios, y=company_vals_large,
        marker_color='#1976d2', showlegend=False,
    ), row=1, col=2)
    fig_bar.add_trace(go.Bar(
        name=industry_label, x=large_ratios, y=industry_vals_large,
        marker_color='#9e9e9e', showlegend=False,
    ), row=1, col=2)
    fig_bar.update_layout(
        barmode='group', template='plotly_white', height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart D: Predicted vs Actual ROA ──
    st.subheader("Model Performance — Actual vs Predicted Mean ROA (Test Years)")
    fig_roa = go.Figure()
    fig_roa.add_trace(go.Scatter(
        x=predictions_by_year['fyear'], y=predictions_by_year['actual_roa_next'],
        mode='lines+markers', name='Actual Mean ROA',
        line=dict(color='#2e7d32'),
    ))
    fig_roa.add_trace(go.Scatter(
        x=predictions_by_year['fyear'], y=predictions_by_year['predicted_roa_next'],
        mode='lines+markers', name='Predicted Mean ROA',
        line=dict(color='#c62828', dash='dash'),
    ))
    fig_roa.update_layout(
        xaxis_title='Fiscal Year', yaxis_title='Mean ROA',
        template='plotly_white', height=400,
    )
    st.plotly_chart(fig_roa, use_container_width=True)

    # ── Chart E: Pie ──
    st.subheader(f"{industry_label} — ROA Improvement Rate")
    fig_pie = go.Figure(data=[go.Pie(
        labels=['ROA Improved', 'ROA Declined'],
        values=[improve_pct, decline_pct],
        marker=dict(colors=['#2e7d32', '#c62828']),
        hole=0.4,
    )])
    fig_pie.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.caption(
        "BA870/AC820 Financial Analytics — Spring 2026 | "
        "Team: Arshdeep Singh Oberoi, Mokhinur Talibzhanova, Steven Marathias | "
        "Data: WRDS Compustat (training) + Yahoo Finance (live)"
    )
