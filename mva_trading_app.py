import streamlit as st
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import datetime
import yfinance as yf  # Fallback for historical if Kite limits hit
import gspread
import json
import re
from urllib.parse import urlparse, parse_qs

# ============================
# CONFIG (Add to Streamlit secrets or env vars)
# ============================
def get_cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        return str(st.secrets[key]).strip()
    return os.environ.get(key, default)

GSHEET_ID = get_cfg("GSHEET_ID")
TOKENSTORE_TAB = get_cfg("TOKENSTORE_TAB", "Sheet1")
TOKENLOG_TAB = get_cfg("TOKENLOG_TAB", "AccessTokenLog")
GCP_SERVICE_ACCOUNT_JSON = get_cfg("GCP_SERVICE_ACCOUNT_JSON")

# Monsoon params (from original MVA)
monsoon_months = [6, 7, 8, 9]
dev_threshold = 8  # 2025 deviation +8% from IMD; update as needed
deviations = {2025: 8, 2026: 0}  # Add current year or fetch dynamically

# ============================
# HELPERS (Adapted from your code)
# ============================
IST = datetime.timezone(datetime.timedelta(hours=5.5), 'IST')

def ist_now() -> datetime.datetime:
    return datetime.datetime.now(tz=IST)

def ist_now_str() -> str:
    return ist_now().strftime("%Y-%m-%d %H:%M:%S")

def ts_is_today_ist(ts: str) -> bool:
    try:
        d = pd.to_datetime(ts, errors="coerce")
        if pd.isna(d):
            return False
        d = d.tz_localize(IST) if d.tzinfo is None else d.tz_convert(IST)
        return d.date() == ist_now().date()
    except:
        return False

def load_service_account_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'("private_key"\s*:\s*")(.+?)(")', lambda m: m.group(1) + m.group(2).replace("\n", "\\n") + m.group(3), raw, flags=re.S)
        return json.loads(fixed)

def get_gspread_client() -> gspread.Client:
    if not GCP_SERVICE_ACCOUNT_JSON:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON.")
    sa_dict = load_service_account_json(GCP_SERVICE_ACCOUNT_JSON)
    return gspread.service_account_from_dict(sa_dict)

@st.cache_data(ttl=10, show_spinner=False)
def read_tokenstore_from_sheet() -> dict:
    out = {"api_key": None, "api_secret": None, "access_token": None, "ts": None, "status": "skipped", "error": ""}
    if not GSHEET_ID:
        out["error"] = "Missing GSHEET_ID."
        return out
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(GSHEET_ID)
        ws = sh.worksheet(TOKENSTORE_TAB)
        row = ws.get("A1:D1")
        if not row or not row[0]:
            out["status"] = "empty"
            return out
        vals = row[0] + [""] * (4 - len(row[0]))
        out["api_key"] = vals[0].strip() or None
        out["api_secret"] = vals[1].strip() or None
        out["access_token"] = vals[2].strip() or None
        out["ts"] = vals[3].strip() or None
        out["status"] = "ok"
        return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out

def write_token_to_sheets_and_verify(access_token: str) -> dict:
    if not GSHEET_ID:
        raise RuntimeError("GSHEET_ID missing.")
    gc = get_gspread_client()
    sh = gc.open_by_key(GSHEET_ID)
    ws_store = sh.worksheet(TOKENSTORE_TAB) if TOKENSTORE_TAB in [w.title for w in sh.worksheets()] else sh.add_worksheet(TOKENSTORE_TAB, rows=1, cols=4)
    ts = ist_now_str()
    ws_store.update("C1:D1", [[access_token, ts]])
    back = ws_store.get("C1:D1")[0]
    if back[0].strip() != access_token.strip():
        raise RuntimeError("Sheet write verify failed.")
    ws_log = sh.worksheet(TOKENLOG_TAB) if TOKENLOG_TAB in [w.title for w in sh.worksheets()] else sh.add_worksheet(TOKENLOG_TAB, rows=100, cols=4)
    ws_log.append_row([ts, access_token, "MVA App", ""])
    read_tokenstore_from_sheet.clear()
    return {"written_ts": ts}

def extract_request_token(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if "request_token=" not in t and "://" not in t:
        return t if len(t) >= 8 else None
    if "request_token=" in t and "://" not in t:
        t = "?" + t if not t.startswith("?") else t
        qs = parse_qs(t[1:])
        return qs.get("request_token", [None])[0]
    try:
        u = urlparse(t)
        qs = parse_qs(u.query)
        return qs.get("request_token", [None])[0]
    except:
        return None

# ============================
# APP
# ============================
st.title("Monsoon Volatility Arbitrage (MVA) Dashboard")
st.markdown("Pairs trading strategy for Indian markets. Authenticate, select pair, and monitor signals.")

# Sidebar: Auth (Inspired by your flow)
with st.sidebar:
    st.header("🔐 Zerodha Login")
    sheet = read_tokenstore_from_sheet()
    if sheet["status"] == "ok":
        st.success("TokenStore OK ✅")
    else:
        st.error(f"TokenStore issue: {sheet['error']}")

    api_key = sheet.get("api_key") or st.text_input("API Key")
    api_secret = sheet.get("api_secret") or st.text_input("API Secret", type="password")

    kite = KiteConnect(api_key=api_key) if api_key else None
    if kite:
        st.markdown(f"[Login to Zerodha]({kite.login_url()})")

    st.text_area("Paste Redirect URL", key="pasted_redirect_url", height=80)
    if st.button("Generate Token from URL"):
        rt = extract_request_token(st.session_state.get("pasted_redirect_url", ""))
        if rt and api_secret:
            try:
                sess = kite.generate_session(rt, api_secret=api_secret)
                access_token = sess["access_token"]
                st.session_state["kite_access_token"] = access_token
                write_token_to_sheets_and_verify(access_token)
                st.success("Token generated and synced ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    # Load from session/sheet
    access_token = st.session_state.get("kite_access_token")
    if not access_token and sheet.get("access_token") and (ts_is_today_ist(sheet["ts"]) or not sheet["ts"]):
        access_token = sheet["access_token"]
        st.session_state["kite_access_token"] = access_token

    if access_token:
        try:
            kite.set_access_token(access_token)
            prof = kite.profile()
            st.success(f"Authenticated as {prof['user_name']}")
        except Exception as e:
            st.error(f"Invalid token: {e}")
            st.session_state.pop("kite_access_token", None)

# Main app (rest same as before, but with auth guard)
if not access_token:
    st.warning("Authenticate first.")
    st.stop()

# Pair selection
stock_a = st.text_input("Stock A (e.g., HINDUNILVR.NS)", "HINDUNILVR.NS")
stock_b = st.text_input("Stock B (e.g., GODREJCP.NS)", "GODREJCP.NS")

# Fetch instruments
instruments = kite.instruments("NSE")
inst_df = pd.DataFrame(instruments)

try:
    token_a = inst_df[inst_df['tradingsymbol'] == stock_a.replace('.NS', '')]['instrument_token'].values[0]
    token_b = inst_df[inst_df['tradingsymbol'] == stock_b.replace('.NS', '')]['instrument_token'].values[0]
except:
    st.error("Invalid stocks.")
    st.stop()

# Historical data
today = datetime.date.today()
from_date = today - datetime.timedelta(days=180)
hist_a = kite.historical_data(token_a, from_date, today, "day")
hist_b = kite.historical_data(token_b, from_date, today, "day")

df_a = pd.DataFrame(hist_a).set_index('date')['close'].rename('A')
df_b = pd.DataFrame(hist_b).set_index('date')['close'].rename('B')
df = pd.concat([df_a, df_b], axis=1).dropna()

# Cointegration, spread, z-score (same as before)
model = sm.OLS(df['A'], sm.add_constant(df['B'])).fit()
residuals = model.resid
adf = adfuller(residuals)
st.write(f"ADF p-value: {adf[1]:.4f}")

df['spread'] = df['A'] - model.params[1] * df['B']
df['mean_spread'] = df['spread'].rolling(30).mean()
df['std_spread'] = df['spread'].rolling(30).std()
df['z_score'] = (df['spread'] - df['mean_spread']) / df['std_spread']

# Signal
now = datetime.datetime.now()
month = now.month
year = now.year
dev = deviations.get(year, 0)
signal = 0
if month in monsoon_months and dev > dev_threshold:
    current_z = df['z_score'].iloc[-1]
    if current_z > 2:
        signal = -1  # Short A, long B
    elif current_z < -2:
        signal = 1   # Long A, short B

# Display (same as before)
st.subheader("Current Metrics")
quotes = kite.quote([f"NSE:{stock_a.replace('.NS', '')}", f"NSE:{stock_b.replace('.NS', '')}"])
col1, col2 = st.columns(2)
col1.write(f"{stock_a} LTP: {quotes[f'NSE:{stock_a.replace('.NS', '')}']['last_price']}")
col2.write(f"{stock_b} LTP: {quotes[f'NSE:{stock_b.replace('.NS', '')}']['last_price']}")
st.write(f"Z-Score: {df['z_score'].iloc[-1]:.2f}")
st.write(f"Signal: {'Buy A / Sell B' if signal == 1 else 'Sell A / Buy B' if signal == -1 else 'No Signal'}")

# Backtest summary
df['returns'] = np.where(df['z_score'].shift(1) > 2, -(df['A'].pct_change() - df['B'].pct_change()),
                         np.where(df['z_score'].shift(1) < -2, df['A'].pct_change() - df['B'].pct_change(), 0))
cum_returns = (1 + df['returns']).cumprod().iloc[-1] - 1
st.write(f"Backtest Total Return (6 months): {cum_returns:.2%}")

# Order placement (same)
if signal != 0:
    qty = st.number_input("Quantity", min_value=1, value=10)
    if st.button("Place Order"):
        try:
            if signal == 1:
                kite.place_order(variety=kite.VARIETY_REGULAR, exchange="NSE", tradingsymbol=stock_a.replace('.NS', ''),
                                 transaction_type=kite.TRANSACTION_TYPE_BUY, quantity=qty, product=kite.PRODUCT_MIS, order_type=kite.ORDER_TYPE_MARKET)
                kite.place_order(variety=kite.VARIETY_REGULAR, exchange="NSE", tradingsymbol=stock_b.replace('.NS', ''),
                                 transaction_type=kite.TRANSACTION_TYPE_SELL, quantity=qty, product=kite.PRODUCT_MIS, order_type=kite.ORDER_TYPE_MARKET)
            else:
                kite.place_order(variety=kite.VARIETY_REGULAR, exchange="NSE", tradingsymbol=stock_a.replace('.NS', ''),
                                 transaction_type=kite.TRANSACTION_TYPE_SELL, quantity=qty, product=kite.PRODUCT_MIS, order_type=kite.ORDER_TYPE_MARKET)
                kite.place_order(variety=kite.VARIETY_REGULAR, exchange="NSE", tradingsymbol=stock_b.replace('.NS', ''),
                                 transaction_type=kite.TRANSACTION_TYPE_BUY, quantity=qty, product=kite.PRODUCT_MIS, order_type=kite.ORDER_TYPE_MARKET)
            st.success("Orders placed!")
        except Exception as e:
            st.error(f"Failed: {e}")
