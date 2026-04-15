# Financial Performance Predictor

**BA870/AC820 Financial & Accounting Analytics — Spring 2026**

Team: Arshdeep Singh Oberoi, Mokhinur Talibzhanova, Steven Marathias

## Overview

A financial analytics dashboard that predicts whether a company's Return on Assets (ROA) will improve or decline in the next fiscal year. The models are trained on WRDS Compustat data (2010–2024) and deployed with live Yahoo Finance data.

## How It Works

1. **Training (offline):** Models were trained on ~155K company-year observations from WRDS Compustat using Random Forest classifiers and regressors with walk-forward validation.
2. **Deployment (this app):** The pre-trained models are loaded from `deployment_artifacts.pkl`. When a user enters a ticker, the app pulls live financials from Yahoo Finance, computes the same features used in training, and generates a prediction.

## Features

- **Prediction Card** — IMPROVE/DECLINE forecast with confidence score
- **Historical Price Chart** — Interactive line chart with 1M/6M/1Y/5Y range buttons
- **Volume vs Return Scatter** — Relationship between trading volume and daily returns
- **Ratios vs Industry** — Company financial ratios compared to SIC industry medians
- **Model Performance** — Actual vs predicted ROA over test years
- **Industry Pie Chart** — ROA improvement rate for similar companies

## Files

- `streamlit_app.py` — Main Streamlit application
- `deployment_artifacts.pkl` — Pre-trained models and pre-computed industry stats
- `requirements.txt` — Python dependencies

## Data Sources

- **Training:** WRDS Compustat Fundamentals Annual (2010–2024)
- **Live:** Yahoo Finance API via `yfinance`
