"""
BA870/AC820 — Financial Analytics Team Project
Streamlit Dashboard: Stock Performance Predictor

Team: Arshdeep Singh Oberoi, Mokhinur Talibzhanova, Steven Marathias

This app trains models on startup from pre-computed features (derived from
WRDS Compustat in the offline Colab notebook) and uses live Yahoo Finance
data to predict whether a company's ROA will improve or decline next year.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Performance Predictor",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────
# Train models on startup (cached — only runs once)
# ──────────────────────────────────────────────
FEATURES = [
    'Debt_ratio', 'PS_ratio', 'log_at', 'ROA_lagged_1_year',
    'PE_ratio', 'ROE', 'earnings_growth',
]

@st.cache_resource
def train_models():
    """Load pre-computed features and train models. Runs once on app startup."""
    csv_path = os.path.join(os.path.dirname(__file__), 'training_features.csv')
    df = pd.read_csv(csv_path)

    # Time-based split (same as Section 2 in Colab)
    fiscal_years = sorted(df['fyear'].dropna().astype(int).unique().tolist())
    split_idx = max(1, int(len(fiscal_years) * 0.80))
    train_years = fiscal_years[:split_idx]
    test_years = fiscal_years[split_idx:]

    train_df = df[df['fyear'].isin(train_years)].copy()
    test_df = df[df['fyear'].isin(test_years)].copy()

    # Clip outliers using train-only quantiles
    clip_bounds = {}
    for feat in FEATURES:
        lo, hi = train_df[feat].quantile([0.01, 0.99])
        clip_bounds[feat] = (float(lo), float(hi))
        train_df.loc[:, feat] = train_df[feat].clip(lo, hi)
        test_df.loc[:, feat] = test_df[feat].clip(lo, hi)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]
    y_train_clf = train_df['target']
    y_test_clf = test_df['target']
    y_train_reg = train_df['ROA_next']
    y_test_reg = test_df['ROA_next']

    # Train classifier
    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=80, max_depth=8, min_samples_leaf=30,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1,
        )),
    ])
    clf.fit(X_train, y_train_clf)

    # Train regressor
    reg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(
            n_estimators=80, max_depth=8, min_samples_leaf=30,
            random_state=42, n_jobs=-1,
        )),
    ])
    reg.fit(X_train, y_train_reg)

    # Test metrics
    clf_pred = clf.predict(X_test)
    clf_ba = balanced_accuracy_score(y_test_clf, clf_pred)
    reg_pred = reg.predict(X_test)
    reg_mae = mean_absolute_error(y_test_reg, reg_pred)

    # Predictions by year (for Chart D)
    test_probs = clf.predict_proba(X_test)[:, 1]
    predictions_by_year = (
        test_df[['fyear', 'ROA_next']]
        .assign(predicted_roa_next=reg_pred, prob_improve=test_probs)
        .groupby('fyear', as_index=False)
        .agg(
            actual_roa_next=('ROA_next', 'mean'),
            predicted_roa_next=('predicted_roa_next', 'mean'),
            mean_improve_probability=('prob_improve', 'mean'),
        )
    )

    # Pre-compute industry stats by 2-digit SIC
    latest_year = df['fyear'].max()
    recent = df[df['fyear'] >= latest_year - 2].copy()
    recent['sich_2d'] = (recent['sich'] // 100).astype('Int64')

    industry_stats_by_sic = {}
    for sic_2d, grp in recent.groupby('sich_2d'):
        if len(grp) >= 30:
            industry_stats_by_sic[int(sic_2d)] = {
                'ROA': float(grp['ROA'].median()) if 'ROA' in grp.columns else float(grp['Debt_ratio'].median()),
                'Debt_ratio': float(grp['Debt_ratio'].median()),
                'PS_ratio': float(grp['PS_ratio'].median()),
                'PE_ratio': float(grp['PE_ratio'].median()),
                'ROE': float(grp['ROE'].median()),
                'improve_pct': float(grp['target'].mean() * 100),
            }

    # Compute ROA median from training data for industry stats
    recent_roa = recent['ROA_next'] if 'ROA_next' in recent.columns else None

    overall_stats = {
        'ROA': float(recent['Debt_ratio'].median()),  # placeholder
        'Debt_ratio': float(recent['Debt_ratio'].median()),
        'PS_ratio': float(recent['PS_ratio'].median()),
        'PE_ratio': float(recent['PE_ratio'].median()),
        'ROE': float(recent['ROE'].median()),
        'improve_pct': float(recent['target'].mean() * 100),
    }

    # Feature medians for fallback
    feature_medians = {
        'earnings_growth': float(df['earnings_growth'].median()),
        'ROA_lagged_1_year': float(df['ROA_lagged_1_year'].median()),
    }

    return {
        'clf': clf,
        'reg': reg,
        'clip_bounds': clip_bounds,
        'clf_ba': clf_ba,
        'reg_mae': reg_mae,
        'predictions_by_year': predictions_by_year,
        'industry_stats_by_sic': industry_stats_by_sic,
        'overall_stats': overall_stats,
        'feature_medians': feature_medians,
    }

try:
    models = train_models()
except Exception as e:
    st.error(f"Could not load training data: {e}")
    st.info("Make sure `training_features.csv` is in the same directory as the app.")
    st.stop()

clf = models['clf']
reg = models['reg']
clip_bounds = models['clip_bounds']
feature_medians = models['feature_medians']
industry_stats_by_sic = models['industry_stats_by_sic']
overall_stats = models['overall_stats']
predictions_by_year = models['predictions_by_year']


# ──────────────────────────────────────────────
# Helper: get price data
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_price_data(ticker):
    try:
        df = yf.download(ticker, start='2010-01-01', auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
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
# Helper: predict performance
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def predict_performance(ticker):
    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return None

    info = {}
    bs = None
    fs = None
    try:
        info = tk.info or {}
    except Exception:
        pass
    try:
        bs = tk.balance_sheet
    except Exception:
        pass
    try:
        fs = tk.financials
    except Exception:
        pass

    if not info and (bs is None or bs.empty) and (fs is None or fs.empty):
        try:
            fi = tk.fast_info
            info = {'marketCap': getattr(fi, 'market_cap', None), 'shortName': ticker.upper()}
        except Exception:
            return None

    def safe_bs(row_name):
        if bs is not None and not bs.empty and row_name in bs.index:
            vals = bs.loc[row_name].dropna()
            if len(vals) > 0:
                return float(vals.iloc[0])
        return None

    def safe_fs(row_name):
        if fs is not None and not fs.empty and row_name in fs.index:
            vals = fs.loc[row_name].dropna()
            if len(vals) > 0:
                return float(vals.iloc[0])
        return None

    total_assets  = safe_bs('Total Assets') or info.get('totalAssets', None)
    total_debt    = safe_bs('Total Debt') or info.get('totalDebt', None)
    net_income    = safe_fs('Net Income') or info.get('netIncomeToCommon', None)
    total_revenue = safe_fs('Total Revenue') or info.get('totalRevenue', None)
    equity        = (safe_bs('Stockholders Equity') or safe_bs('Total Stockholder Equity')
                     or info.get('totalStockholderEquity', None))
    market_cap    = info.get('marketCap', None)
    if market_cap is None:
        try:
            market_cap = getattr(tk.fast_info, 'market_cap', None)
        except Exception:
            pass
    trailing_pe   = info.get('trailingPE', None)
    sector        = info.get('sector', 'Unknown')
    company_name  = info.get('shortName', ticker.upper())
    sic_code      = info.get('sic', None)

    if total_assets is None and total_revenue is None and net_income is None:
        return None

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
        'Debt_ratio': Debt_ratio, 'PS_ratio': PS_ratio, 'log_at': log_at,
        'ROA_lagged_1_year': roa_lag_val, 'PE_ratio': PE_ratio,
        'ROE': ROE, 'earnings_growth': eg_val,
    }

    feature_sources = {
        'Debt_ratio': 'actual' if not np.isnan(Debt_ratio) else 'N/A',
        'PS_ratio': 'actual' if not np.isnan(PS_ratio) else 'N/A',
        'log_at': 'actual' if not np.isnan(log_at) else 'N/A',
        'ROA_lagged_1_year': roa_lag_source,
        'PE_ratio': 'actual' if not np.isnan(PE_ratio) else 'N/A',
        'ROE': 'actual' if not np.isnan(ROE) else 'N/A',
        'earnings_growth': eg_source,
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
        'ticker': ticker.upper(), 'company_name': company_name,
        'sector': sector, 'sic_code': sic_code,
        'features': feature_dict, 'feature_sources': feature_sources,
        'feature_clipped': feature_clipped, 'ROA_current': ROA,
        'prediction': 'IMPROVE' if pred_class == 1 else 'DECLINE',
        'confidence': float(max(prob)), 'prob_improve': float(prob[1]),
        'pred_roa_next': float(pred_roa),
    }


# ──────────────────────────────────────────────
# Helper: industry stats
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
    industry_means = {k: stats[k] for k in ['ROA', 'Debt_ratio', 'PS_ratio', 'PE_ratio', 'ROE']}
    return industry_means, stats['improve_pct'], 100 - stats['improve_pct'], industry_label


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

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10,
                           placeholder="e.g. NVDA, TSLA, META")
with col_btn:
    st.write("")
    analyze = st.button("🔍 Analyze", use_container_width=True)

if analyze and ticker.strip():
    ticker = ticker.strip().upper()

    with st.spinner(f"Fetching data for {ticker}..."):
        pred = predict_performance(ticker)
        price_df = get_price_data(ticker)

    if pred is None:
        st.error(f"Could not fetch financial data for **{ticker}**. Check the symbol and try again.")
        st.info("💡 Tip: Try common tickers like AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOGL")
        st.stop()

    if price_df.empty:
        st.warning(f"No price history found for **{ticker}**. Showing prediction only.")

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

    st.caption(f"Based on WRDS Compustat data ({industry_label}) & Random Forest Classifier / Regressor")

    # ── Feature Details ──
    with st.expander("Show model input features"):
        feat_data = []
        for feat in FEATURES:
            val = pred['features'][feat]
            feat_data.append({
                'Feature': feat,
                'Value': f"{val:.4f}" if not np.isnan(val) else "N/A",
                'Source': pred['feature_sources'][feat],
                'Clipped': '✂ yes' if pred['feature_clipped'].get(feat) else '',
            })
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    if not price_df.empty:
        # ── Chart A: Price ──
        st.subheader(f"{ticker} — Historical Close Price")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=price_df.index, y=price_df['Close'],
            mode='lines', name='Close Price', line=dict(color='#1976d2', width=1.5),
        ))
        fig_line.add_trace(go.Scatter(
            x=price_df.index, y=price_df['ma_20'],
            mode='lines', name='20-Day MA', line=dict(color='#ff9800', width=1, dash='dash'),
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
            scatter_df, x='Volume', y='daily_return', opacity=0.4,
            labels={'Volume': 'Trading Volume', 'daily_return': 'Daily Return'},
            template='plotly_white',
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Chart C: Ratios vs Industry ──
    st.subheader(f"{ticker} Ratios vs {industry_label} Median")
    small_ratios = ['ROA', 'Debt_ratio', 'ROE']
    large_ratios = ['PS_ratio', 'PE_ratio']
    company_small = [pred['ROA_current'], pred['features']['Debt_ratio'], pred['features']['ROE']]
    industry_small = [industry_means[r] for r in small_ratios]
    company_large = [pred['features']['PS_ratio'], pred['features']['PE_ratio']]
    industry_large = [industry_means[r] for r in large_ratios]

    fig_bar = make_subplots(rows=1, cols=2,
        subplot_titles=['Profitability & Leverage', 'Valuation Multiples'],
        horizontal_spacing=0.15)
    fig_bar.add_trace(go.Bar(name=ticker, x=small_ratios, y=company_small,
        marker_color='#1976d2'), row=1, col=1)
    fig_bar.add_trace(go.Bar(name=industry_label, x=small_ratios, y=industry_small,
        marker_color='#9e9e9e'), row=1, col=1)
    fig_bar.add_trace(go.Bar(name=ticker, x=large_ratios, y=company_large,
        marker_color='#1976d2', showlegend=False), row=1, col=2)
    fig_bar.add_trace(go.Bar(name=industry_label, x=large_ratios, y=industry_large,
        marker_color='#9e9e9e', showlegend=False), row=1, col=2)
    fig_bar.update_layout(barmode='group', template='plotly_white', height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart D: Actual vs Predicted ROA ──
    st.subheader("Model Performance — Actual vs Predicted Mean ROA (Test Years)")
    fig_roa = go.Figure()
    fig_roa.add_trace(go.Scatter(
        x=predictions_by_year['fyear'], y=predictions_by_year['actual_roa_next'],
        mode='lines+markers', name='Actual Mean ROA', line=dict(color='#2e7d32'),
    ))
    fig_roa.add_trace(go.Scatter(
        x=predictions_by_year['fyear'], y=predictions_by_year['predicted_roa_next'],
        mode='lines+markers', name='Predicted Mean ROA', line=dict(color='#c62828', dash='dash'),
    ))
    fig_roa.update_layout(xaxis_title='Fiscal Year', yaxis_title='Mean ROA',
        template='plotly_white', height=400)
    st.plotly_chart(fig_roa, use_container_width=True)

    # ── Chart E: Pie ──
    st.subheader(f"{industry_label} — ROA Improvement Rate")
    fig_pie = go.Figure(data=[go.Pie(
        labels=['ROA Improved', 'ROA Declined'],
        values=[improve_pct, decline_pct],
        marker=dict(colors=['#2e7d32', '#c62828']), hole=0.4,
    )])
    fig_pie.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.caption(
        "BA870/AC820 Financial Analytics — Spring 2026 | "
        "Team: Arshdeep Singh Oberoi, Mokhinur Talibzhanova, Steven Marathias | "
        "Data: WRDS Compustat (training) + Yahoo Finance (live)"
    )
