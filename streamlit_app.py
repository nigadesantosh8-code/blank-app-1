import io, time
from datetime import datetime
import numpy as np, pandas as pd, requests, streamlit as st, yfinance as yf

st.set_page_config(page_title="NYSE Screener", layout="wide")
st.title("NYSE screener")
H = {"User-Agent": "Mozilla/5.0"}
clean = lambda s: s.strip().upper().replace(".", "-")

@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(name):
    if name == "Nifty 500 (NSE)":
        urls = ["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"]
        for url in urls:
            try:
                txt = requests.get(url, headers=H, timeout=20).text
                df = pd.read_csv(io.StringIO(txt))
                return sorted({str(x).strip().upper() + ".NS" for x in df["Symbol"].dropna()})
            except Exception:
                continue
        raise RuntimeError("Nifty 500 list download nahi hui")
    if name == "S&P 500":
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=H, timeout=20).text
        return sorted({clean(s) for s in pd.read_html(io.StringIO(html))[0]["Symbol"]})
    t = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", headers=H, timeout=20).text
    df = pd.read_csv(io.StringIO(t), sep="|", dtype=str)
    df = df[(df["Exchange"] == "N") & (df["ETF"] == "N") & (df["Test Issue"] == "N")]
    s = df["ACT Symbol"].dropna()
    return sorted({clean(x) for x in s[~s.str.contains(r"[\$\^]")]})

def rsi_series(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)

def harsi(df, n=14, smooth=7):
    o, h, l, c = (rsi_series(df[k], n) - 50 for k in ["Open", "High", "Low", "Close"])
    hc = (o + h + l + c) / 4
    ho = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if np.isnan(hc.iloc[i]) or np.isnan(o.iloc[i]):
            continue
        if i == 0 or np.isnan(ho.iloc[i-1]):
            ho.iloc[i] = (o.iloc[i] + c.iloc[i]) / 2
        else:
            ho.iloc[i] = (ho.iloc[i-1] * (smooth - 1) + (ho.iloc[i-1] + hc.iloc[i-1]) / 2) / smooth
    return float(hc.iloc[-1]), bool(hc.iloc[-1] > ho.iloc[-1])

@st.cache_data(ttl=240, show_spinner=False)
def fetch_metrics(tickers, nonce):
    rows = []
    for i in range(0, len(tickers), 100):
        chunk = list(tickers[i:i+100])
        data = yf.download(chunk, period="3mo", interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        if data is not None and not data.empty:
            have = set(data.columns.get_level_values(0))
            for t in chunk:
                if t not in have: continue
                df = data[t].dropna(subset=["Close"])
                if len(df) < 35: continue
                avg = df["Volume"].iloc[-21:-1].mean()
                if not np.isfinite(avg) or avg <= 0: continue
                hv, hg = harsi(df)
                rows.append(dict(Ticker=t, Price=df["Close"].iloc[-1], AvgVol=avg,
                                 VolRatio=df["Volume"].iloc[-1] / avg,
                                 RSI=float(rsi_series(df["Close"]).iloc[-1]),
                                 HARSI=hv, HARSI_Green=hg))
        time.sleep(1)
    return pd.DataFrame(rows), datetime.now()

@st.cache_data(ttl=21600, show_spinner=False)
def fundamentals(t):
    info = yf.Ticker(t).info
    time.sleep(0.4)
    return dict(Ticker=t, Company=info.get("shortName", t),
                PE=pd.to_numeric(info.get("trailingPE"), errors="coerce"))

with st.sidebar:
    uni = st.selectbox("Stocks", ["Nifty 500 (NSE)", "S&P 500", "NYSE-listed stocks"])
    max_pe = st.slider("P/E below", 1.0, 60.0, 20.0, 0.5)
    min_ratio = st.slider("Volume ratio at least", 1.0, 10.0, 2.0, 0.1)
    min_rsi = st.slider("RSI above", 0, 100, 50)
    need_harsi = st.checkbox("HARSI green (bullish)", value=False)
    min_price = st.number_input("Min price", 0.0, 1000.0, 2.0)
    min_vol = st.number_input("Min avg volume", 0, 50_000_000, 200_000, 50_000)
    top_n = st.slider("Show top", 5, 100, 25, 5)
    auto = st.toggle("Auto-refresh (5 min)", value=False)
    if st.button("Scan now", use_container_width=True):
        st.session_state["n"] = st.session_state.get("n", 0) + 1
nonce = st.session_state.get("n", 0)

def render():
    try:
        tickers = tuple(load_universe(uni))
    except Exception as e:
        st.error(f"Stock list download nahi hui ({e}). Thodi der baad Scan now dabao.")
        return
    with st.spinner(f"{len(tickers)} stocks scan ho rahe hain, 1-3 minute lagte hain..."):
        m, when = fetch_metrics(tickers, nonce)
    if m.empty:
        st.error("Yahoo ne data nahi diya. 1 minute baad Scan now dabao.")
        return
    c = m[(m.VolRatio >= min_ratio) & (m.RSI > min_rsi) & (m.Price >= min_price) & (m.AvgVol >= min_vol)]
    if need_harsi:
        c = c[c.HARSI_Green]
    c = c.sort_values("VolRatio", ascending=False).head(80)
    recs = []
    bar = st.progress(0.0, text="P/E check ho raha hai")
    for i, t in enumerate(c.Ticker, 1):
        try: recs.append(fundamentals(t))
        except Exception: pass
        bar.progress(i / len(c))
    bar.empty()
    st.caption(f"Scanned {len(m)} | Volume+RSI pass {len(c)} | Data time {when:%H:%M:%S}")
    if not recs:
        st.info("Abhi koi stock nahi mila. Sliders dheele karo.")
        return
    r = c.merge(pd.DataFrame(recs), on="Ticker")
    r = r[(r.PE > 0) & (r.PE < max_pe)].head(top_n)
    r = r[["Ticker", "Company", "Price", "PE", "VolRatio", "RSI", "HARSI"]].round(2).reset_index(drop=True)
    r.index += 1
    if r.empty:
        st.info("Teeno shart poori karne wala koi stock nahi. Sliders dheele karo.")
    else:
        st.dataframe(r, use_container_width=True)
    st.caption("Yahoo data ~15 min late. Market ke beech volume adhoora hota hai. Investment salah nahi.")

st.fragment(run_every="5m" if auto else None)(render)()
