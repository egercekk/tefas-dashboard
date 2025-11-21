import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import plotly.express as px

from tefas import Crawler

# ============================================================
# 🔐 PASSWORD PROTECTION
# ============================================================

import hashlib

def _hash_password(pwd: str) -> str:
    """SHA256 hash function"""
    return hashlib.sha256(pwd.encode()).hexdigest()

def check_password():
    """Password gate using Streamlit Secrets"""
    # Secrets içindeki hash'i alıyoruz
    REAL_HASH = st.secrets["auth"]["password_hash"]

    # Session state üzerinde password flag'i yoksa ekle
    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    # Henüz giriş yapılmamışsa login ekranını göster
    if not st.session_state["password_ok"]:
        st.markdown("### 🔒 Access Protection")
        password = st.text_input("Enter the password:", type="password")

        if st.button("Login"):
            if _hash_password(password) == REAL_HASH:
                st.session_state["password_ok"] = True
                st.rerun()  # ❗ Streamlit 1.30+ doğru komut
            else:
                st.error("❌ Wrong password")

        # Şifre yanlışsa veya boşsa uygulamanın geri kalanını durdur
        st.stop()

# ============================================================
# Streamlit UI Başlangıcı
# ============================================================

st.set_page_config(page_title="TEFAS Dashboard", page_icon="📈", layout="wide")

# 🔑 ŞİFRE KONTROLÜ (Uygulamanın devamı bundan SONRA gelir)
check_password()

# ============================================================
# BUNDAN SONRASINA SENİN ORİJİNAL DASHBOARD KODUN GELİYOR
# (CSS, sidebar, macro page, charts vs.)
# ============================================================

# Örnek olarak:
st.markdown(
    """
    <style>
    .stApp { background-color: #020617; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------
# TEFAS DATA FETCH
# ------------------------------------------
def fetch_tefas(codes, start, end, kind):
    crawler = Crawler()
    frames = []
    for code in codes:
        df = crawler.fetch(start=start, end=end, name=code, kind=kind)
        if df is not None and not df.empty:
            df["code"] = code
            frames.append(df)
        else:
            st.warning(
                f"⚠️ No TEFAS data returned for **{code}** in the selected date range. "
                f"Please double-check the fund code, type and date range."
            )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("date")


def fetch_tefas_all(start, end, kind=None):
    """
    Used in Macro tab: fetch all funds for a given type (or all types),
    instead of a single code.
    """
    crawler = Crawler()
    df = crawler.fetch(start=start, end=end, name=None, kind=kind)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values("date")


# ------------------------------------------
# FIXED BIST 50 LIST (NO LXML, NO READ_HTML)
# ------------------------------------------
@st.cache_data
def get_bist50_constituents():
    """
    Static BIST 50 list (no KAP scraping, no lxml dependency).
    Yahoo Finance format (.IS) is included.
    UPDATE THIS LIST MANUALLY IF BORSA CHANGES BIST 50 COMPONENTS.
    """
    return [
        "AEFES.IS", "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS",
        "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "CCOLA.IS", "CIMSA.IS",
        "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "EKGYO.IS", "ENKAI.IS",
        "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS",
        "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS",
        "KOZAL.IS", "KRDMD.IS", "KUYAS.IS", "MAVI.IS", "MGROS.IS",
        "MIATK.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
        "SASA.IS", "SISE.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS",
        "THYAO.IS", "TKFEN.IS", "TSKB.IS", "TTKOM.IS", "TUPRS.IS",
        "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YKBNK.IS"
    ]


@st.cache_data
def fetch_bist50_prices_yf(start_date, end_date):
    """
    Fetch BIST 50 adjusted close prices from yfinance.
    Returns DataFrame: ['date','code','yahoo_code','price_try']
    """
    tickers = get_bist50_constituents()

    if isinstance(start_date, datetime):
        start_dt = start_date
    else:
        start_dt = datetime.combine(start_date, datetime.min.time())

    if isinstance(end_date, datetime):
        end_dt = end_date
    else:
        end_dt = datetime.combine(end_date, datetime.min.time())

    df = yf.download(
        tickers=tickers,
        start=start_dt,
        end=end_dt + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    else:
        prices = df["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.index = prices.index.tz_localize(None)
    prices = prices.dropna(how="all")

    df2 = prices.stack().reset_index()
    df2.columns = ["date", "yahoo_code", "price_try"]
    df2["code"] = df2["yahoo_code"].str.replace(".IS", "", regex=False)

    return df2


@st.cache_data
def fetch_usdtry_series(start_date, end_date):
    """
    Currently not used (currency is fixed to TRY),
    kept here for future USD analysis if needed.
    """
    fx = yf.download(
        "USDTRY=X",
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if fx.empty:
        return pd.DataFrame()

    s = fx["Adj Close"] if "Adj Close" in fx.columns else fx["Close"]
    s.index = s.index.tz_localize(None)
    out = s.reset_index()
    out.columns = ["date", "usdtry"]
    return out


# ------------------------------------------
# NUMBER FORMATTING HELPERS
# ------------------------------------------
def fmt_int(x):
    """Format like 2.411.844.935"""
    try:
        s = f"{int(float(x)):,}"
        return s.replace(",", ".")
    except Exception:
        return x


def fmt_percent(x):
    """7.28 -> 7.28%"""
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return x


def fmt_float(x):
    """8.640000 -> 8.64"""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x


# ------------------------------------------
# MOVE ALL-ZERO NUMERIC COLUMNS TO THE END
# ------------------------------------------
def move_zero_columns_last(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    zero_cols = [c for c in numeric_cols if df[c].fillna(0).eq(0).all()]
    new_cols = [c for c in df.columns if c not in zero_cols] + zero_cols
    return df[new_cols]


# ------------------------------------------
# PRETTY DISPLAY NAME FOR COLUMNS
# ------------------------------------------
def pretty(col_name: str) -> str:
    return col_name.replace("_", " ").title()


# ------------------------------------------
# SIMPLE BINOMIAL HELPERS (no SciPy)
# ------------------------------------------
def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(binom_pmf(i, n, p) for i in range(0, k + 1))


# ------------------------------------------
# STREAMLIT DASHBOARD
# ------------------------------------------
st.set_page_config(page_title="TEFAS Funds", page_icon="📈", layout="wide")

# --------- GLOBAL STYLE (DARK + LEFT MENU) ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #020617; }
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #111827;
        padding-top: 0;
    }
    section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }

    .side-section-title {
        font-size: 0.70rem;
        letter-spacing: 0.08em;
        color: #9ca3af;
        text-transform: uppercase;
        margin-top: 1.2rem;
        margin-bottom: 0.15rem;
    }

    div[data-testid="stRadio"] > label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.1rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0px; }
    div[data-testid="stRadio"] button {
        background-color: transparent;
        border-radius: 0.35rem;
        padding: 0.35rem 0.5rem;
        color: #e5e7eb;
        font-size: 0.86rem;
        width: 100%;
        justify-content: flex-start;
        border: 0px;
    }
    div[data-testid="stRadio"] button[aria-checked="true"] {
        background: linear-gradient(90deg, #0f172a, #111827);
        border-left: 3px solid #3b82f6;
        color: #ffffff;
    }
        div[data-testid="stRadio"] div[role="radiogroup"] > label {
        margin-bottom: 60px;
    }

    div[data-testid="stRadio"] button {
        margin-bottom: 60px;
    }
    
    div[data-testid="stRadio"] button[aria-checked="false"]:hover {
        background-color: #030712;
    }

    div[data-testid="stMetric"] {
        background-color: #020617;
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #1f2937;
        text-align: center;
    }
    div[data-testid="stMetric"] > label {
        font-size: 0.75rem;
        color: #9CA3AF;
    }
    div[data-testid="stMetric"] > div {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F9FAFB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- LANGUAGE STATE & HELPER (senin kodun) ----------
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

def _t(tr: str, en: str) -> str:
    return en

# --------- MENU STATE ----------
if "active_group" not in st.session_state:
    st.session_state["active_group"] = "tefas"  # başlangıç TEFAS
if "tefas_sections" not in st.session_state:
    st.session_state["tefas_sections"] = "macro"
if "stocks_sections" not in st.session_state:
    st.session_state["stocks_sections"] = "arb"

# --------- SIDEBAR (LEFT MENU) ----------
with st.sidebar:
    # 1) Ana grup seçimi
    main_group = st.radio(
        "Main Group",
        ["tefas", "stocks"],
        index=0,
        format_func=lambda x: "📈 TEFAS Funds" if x == "tefas" else "📊 STOCKS",
        key="main_group",
    )

    # 2) Grup içi alt sekmeler
    if main_group == "tefas":
        main_page = st.radio(
            "TEFAS Sections",
            ["macro", "raw", "deltas", "charts", "compare", "stats", "prob", "research"],
            label_visibility="collapsed",
            key="tefas_sections",
            format_func=lambda x: {
                "macro": "🌐 MACRO",
                "raw": "📄 RAW DATA",
                "deltas": "📌 DELTAS",
                "charts": "📊 CHARTS",
                "compare": "🔍 COMPARE",
                "stats": "📈 STATISTICS",
                "prob": "🎲 PROBABILITY",
                "research": "📚 RESEARCH",
            }[x],
        )
    else:  # main_group == "stocks"
        main_page = st.radio(
            "Stock Sections",
            ["arb"],
            label_visibility="collapsed",
            key="stocks_sections",
            format_func=lambda x: "🔁 BIST 50 ARBITRAGE",
        )


# --------- MAIN TITLE ----------
# Arbitraj sekmesinde başlık olmasın
if main_page != "arb":
    st.markdown(
        "<h1 style='text-align: center;'>📈 TEFAS Funds</h1>",
        unsafe_allow_html=True,
    )


# ==========================================
# INPUTS
# ==========================================

# Fund type / codes
if main_page == "macro":
    # MACRO sekmesinde eski davranış aynen kalsın
    kind = st.selectbox("Fund Type", ["YAT", "EMK", "BYF"], key="kind_macro")
    codes_text = ""  # not used in macro

elif main_page == "arb":
    # ARBITRAGE sekmesi için:
    # - Fund Type GÖZÜKMEZ
    # - Label "BIST 50 (comma separated)" olur
    codes_text = st.text_input(
        "BIST 50 (comma separated):",
        "AEFES, AKBNK",
        key="codes_arb",
    )
    # kind TEFAS tarafında kullanılmayacak ama altta fonksiyon imzası için
    # yine de bir değer verelim ki hata çıkmasın.
    kind = "YAT"   # dummy / default, tamamen görünmez, sadece teknik

else:
    # Diğer sekmeler (raw, deltas, charts, compare, stats, prob, research)
    # için eski davranış:
    codes_text = st.text_input(
        "Fund Codes (comma separated):",
        "SPN",
        key="codes_text",
    )
    kind = st.selectbox("Fund Type", ["YAT", "EMK", "BYF"], key="kind_normal")

# If we are in Research tab: show only link page, stop other logic
if main_page == "research":
    st.subheader("📚 Research – Quick Links")

    st.markdown(
        "Use the links below to quickly open the official TEFAS fund / data pages."
    )

    st.markdown(
        """
- [TEFAS - Fund Information](https://www.tefas.gov.tr/)
        """
    )

    codes_for_links = [c.strip().upper() for c in codes_text.split(",") if c.strip()]
    if codes_for_links:
        st.markdown("---")
        st.markdown("Quick links for entered fund codes:")
        for code in sorted(set(codes_for_links)):
            tefas_url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={code}"
            st.markdown(f"- **{code}** → [TEFAS]({tefas_url})")

    st.stop()

# Dates
col1, col2, col3 = st.columns([1, 1, 1.2])

default_start = datetime(2025, 10, 19)
default_end = datetime(2025, 11, 18)

start = col1.date_input("Start Date", default_start)
end = col2.date_input("End Date", default_end)

# Quick range
quick_range_options = [
    ("custom", "Custom"),
    ("ytd", "Year-to-date (YTD)"),
    ("3m", "Last 3 Months"),
    ("6m", "Last 6 Months"),
    ("1y", "Last 1 Year"),
]

quick_range_label = col3.selectbox(
    "Quick Range",
    [lbl for key, lbl in quick_range_options],
    index=0,
    help="Choose a predefined date range or keep 'Custom' to use the selected start/end dates.",
)

quick_range_key = [k for k, lbl in quick_range_options if lbl == quick_range_label][0]

# ------------------------------------------
# FETCH BUTTON
# ------------------------------------------
if st.button("📥 Fetch Data"):
    effective_start = start
    effective_end = end
    today = datetime.today().date()

    if quick_range_key != "custom":
        if quick_range_key == "ytd":
            effective_start = datetime(today.year, 1, 1).date()
        elif quick_range_key == "3m":
            effective_start = today - timedelta(days=90)
        elif quick_range_key == "6m":
            effective_start = today - timedelta(days=180)
        elif quick_range_key == "1y":
            effective_start = today - timedelta(days=365)
        effective_end = today

    if effective_start > effective_end:
        st.error(
            "❌ Invalid date range: **Start Date** must be earlier than **End Date**. "
            "Please adjust the dates or the quick filter."
        )

    else:
        # ======================================================
        # 1) MACRO TAB (TEFAS)
        # ======================================================
        if main_page == "macro":
            macro_df = fetch_tefas_all(
                str(effective_start), str(effective_end), kind=kind
            )

            if macro_df.empty:
                st.error(
                    "❌ No TEFAS data returned for the selected date range and type. "
                    "Try expanding the date range or changing the fund type."
                )
            else:
                macro_df["date"] = pd.to_datetime(macro_df["date"])
                macro_df = move_zero_columns_last(macro_df)

                last_day = macro_df["date"].max()
                latest = macro_df[macro_df["date"] == last_day].copy()

                n_funds = latest["code"].nunique() if "code" in latest.columns else None

                if "market_cap" in latest.columns:
                    latest["market_cap"] = pd.to_numeric(
                        latest["market_cap"], errors="coerce"
                    )
                    total_aum = latest["market_cap"].sum()
                else:
                    total_aum = None

                col_m1, col_m2 = st.columns(2)
                if n_funds is not None:
                    col_m1.metric("Number of Funds", fmt_int(n_funds))
                if total_aum is not None:
                    col_m2.metric("Total AUM (Last Day)", fmt_int(total_aum))

                st.markdown("### Top & Bottom 20 Funds by AUM (Last Day)")

                if "market_cap" in latest.columns:
                    top20 = latest.sort_values("market_cap", ascending=False).head(20)
                    bottom20 = latest.sort_values("market_cap", ascending=True).head(20)

                    col_top, col_bottom = st.columns(2)

                    with col_top:
                        st.caption("Top 20 Funds")
                        fig_top = px.bar(
                            top20,
                            x="code",
                            y="market_cap",
                            labels={"code": "Fund", "market_cap": "Market Cap"},
                        )
                        st.plotly_chart(fig_top, use_container_width=True)

                    with col_bottom:
                        st.caption("Bottom 20 Funds")
                        fig_bottom = px.bar(
                            bottom20,
                            x="code",
                            y="market_cap",
                            labels={"code": "Fund", "market_cap": "Market Cap"},
                        )
                        st.plotly_chart(fig_bottom, use_container_width=True)

                st.markdown("---")
                st.subheader("Macro Raw Data")

                macro_df["date"] = macro_df["date"].dt.strftime("%Y-%m-%d")

                percent_cols_all = [
                    "other",
                    "government_bond",
                    "commercial_paper",
                    "stock",
                    "government_lease_certificates_tl",
                    "private_sector_bond",
                    "repo",
                    "asset_backed_securities",
                    "futures_cash_collateral",
                    "foreign_investment_fund_participation_shares",
                ]
                percent_cols_macro = [
                    c for c in percent_cols_all if c in macro_df.columns
                ]

                int_cols_all = [
                    "market_cap",
                    "number_of_shares",
                    "number_of_investors",
                ]
                int_cols_macro = [c for c in int_cols_all if c in macro_df.columns]

                float_cols_macro = [
                    c
                    for c in macro_df.columns
                    if c not in int_cols_macro + percent_cols_macro
                    and pd.api.types.is_numeric_dtype(macro_df[c])
                ]

                display_map_macro = {col: pretty(col) for col in macro_df.columns}
                macro_display = macro_df.rename(columns=display_map_macro)

                macro_format_map = {}
                for col in int_cols_macro:
                    macro_format_map[pretty(col)] = fmt_int
                for col in percent_cols_macro:
                    macro_format_map[pretty(col)] = fmt_percent
                for col in float_cols_macro:
                    macro_format_map[pretty(col)] = fmt_float

                macro_styled = macro_display.style.format(macro_format_map)
                st.dataframe(macro_styled, hide_index=True)

        # ======================================================
        # 2) OTHER TABS (TEFAS) + ARBITRAGE (BIST50)
        # ======================================================
        else:
            # ---------------- TEFAS DATA (other tabs) ----------------
            codes = [c.strip().upper() for c in codes_text.split(",") if c.strip()]
            data = fetch_tefas(codes, str(effective_start), str(effective_end), kind)

            # ARBITRAGE tabında TEFAS verisi kullanılmıyor,
            # ama diğer bütün tablarda kullanılmaya devam ediyor.

            if main_page != "arb":
                if data.empty:
                    st.error(
                        "❌ No data returned for the selected funds and date range. "
                        "Try expanding the date range or confirming the fund codes and type.",
                    )
                    st.stop()

                data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
                data = move_zero_columns_last(data)

                percent_cols_all = [
                    "other",
                    "government_bond",
                    "commercial_paper",
                    "stock",
                    "government_lease_certificates_tl",
                    "private_sector_bond",
                    "repo",
                    "asset_backed_securities",
                    "futures_cash_collateral",
                    "foreign_investment_fund_participation_shares",
                ]
                percent_cols = [c for c in percent_cols_all if c in data.columns]

                int_cols_all = [
                    "market_cap",
                    "number_of_shares",
                    "number_of_investors",
                ]
                int_cols = [c for c in int_cols_all if c in data.columns]

                float_cols = [
                    c
                    for c in data.columns
                    if c not in int_cols + percent_cols
                    and pd.api.types.is_numeric_dtype(data[c])
                ]

                delta_cols = int_cols + percent_cols + float_cols

            # -------- RAW DATA PAGE --------
            if main_page == "raw":
                st.subheader("🧾 Raw Data")

                display_map_raw = {col: pretty(col) for col in data.columns}
                data_display = data.rename(columns=display_map_raw)

                raw_format_map = {}
                for col in int_cols:
                    raw_format_map[pretty(col)] = fmt_int
                for col in percent_cols:
                    raw_format_map[pretty(col)] = fmt_percent
                for col in float_cols:
                    raw_format_map[pretty(col)] = fmt_float

                data_styled = data_display.style.format(raw_format_map)
                st.dataframe(data_styled, hide_index=True)

            # -------- DELTAS PAGE --------
            elif main_page == "deltas":
                st.subheader("📌 Last Values (Δ vs Previous Day)")

                delta_df = data.copy()
                for c in delta_cols:
                    delta_df[c] = pd.to_numeric(delta_df[c], errors="coerce")

                delta_df = delta_df.sort_values(["code", "date"])
                delta_df[delta_cols] = delta_df.groupby("code")[delta_cols].diff()

                delta_latest = delta_df.groupby("code").tail(1)
                delta_latest = move_zero_columns_last(delta_latest)

                display_map_delta = {}
                for col in delta_latest.columns:
                    if col in delta_cols:
                        display_map_delta[col] = "Δ " + pretty(col)
                    else:
                        display_map_delta[col] = pretty(col)

                delta_display = delta_latest.rename(columns=display_map_delta)

                delta_format_map = {}
                for col in int_cols:
                    delta_format_map["Δ " + pretty(col)] = fmt_int
                for col in percent_cols:
                    delta_format_map["Δ " + pretty(col)] = fmt_percent
                for col in float_cols:
                    delta_format_map["Δ " + pretty(col)] = fmt_float

                delta_styled = delta_display.style.format(delta_format_map)
                st.dataframe(delta_styled, hide_index=True)

            # -------- CHARTS PAGE --------
            elif main_page == "charts":
                st.subheader("📊 Price Chart")

                if "price" in data.columns:
                    fig_price = px.line(data, x="date", y="price", color="code")
                    fig_price.update_layout(
                        title="Fund Price Evolution Over Time",
                        xaxis_title="Date",
                        yaxis_title="Price",
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.info(
                        "⚠️ Column **'price'** is not available in the returned dataset. "
                        "Price chart cannot be generated for this selection.",
                    )

                st.markdown("---")
                st.subheader("📊 Asset Allocation – Latest Day")

                if percent_cols:
                    codes_unique = sorted(data["code"].unique())
                    selected_code = st.selectbox(
                        "Select fund for allocation charts",
                        codes_unique,
                    )

                    latest_alloc = (
                        data[data["code"] == selected_code].sort_values("date").tail(1)
                    )

                    alloc_df = latest_alloc[percent_cols].T.reset_index()
                    alloc_df.columns = ["Category", "Value"]
                    alloc_df["Category"] = alloc_df["Category"].apply(pretty)
                    alloc_df = alloc_df.sort_values("Value", ascending=False)

                    st.caption("Pie Chart")
                    fig_pie = px.pie(
                        alloc_df,
                        names="Category",
                        values="Value",
                        title=f"Asset Allocation (%) - {selected_code}",
                        hole=0.35,
                    )
                    fig_pie.update_layout(height=500)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.caption("Treemap (sorted by weight)")
                    fig_tree = px.treemap(
                        alloc_df,
                        path=["Category"],
                        values="Value",
                        title=f"Asset Allocation Treemap - {selected_code}",
                    )
                    fig_tree.update_traces(
                        texttemplate="%{label}<br>%{value:.2f}%",
                        hovertemplate="<b>%{label}</b><br>Share: %{value:.2f}%<extra></extra>",
                    )
                    fig_tree.update_layout(height=600)
                    st.plotly_chart(fig_tree, use_container_width=True)

                else:
                    st.info(
                        "No allocation (percentage) columns are available in the dataset. "
                        "Allocation charts cannot be generated for this selection.",
                    )

            # -------- COMPARE PAGE --------
            elif main_page == "compare":
                st.subheader("🔍 Compare Funds")

                if "price" not in data.columns:
                    st.warning(
                        "⚠️ Column **'price'** is missing. "
                        "The Compare tab requires the price column to calculate returns.",
                    )
                else:
                    all_codes = sorted(data["code"].unique())
                    selected_codes = st.multiselect(
                        "Select funds (at least 1):",
                        all_codes,
                        default=all_codes,
                    )

                    if len(selected_codes) == 0:
                        st.info("Select at least one fund to compare.")
                    else:
                        comp = data[data["code"].isin(selected_codes)].copy()
                        comp["date"] = pd.to_datetime(comp["date"])
                        comp["price"] = pd.to_numeric(comp["price"], errors="coerce")
                        comp = comp.dropna(subset=["price"])

                        last_rows = (
                            comp.sort_values(["code", "date"])
                            .groupby("code")
                            .tail(1)
                        )

                        metric_cols = []
                        if "market_cap" in last_rows.columns:
                            metric_cols.append("market_cap")
                        if "number_of_investors" in last_rows.columns:
                            metric_cols.append("number_of_investors")
                        if "risk" in last_rows.columns:
                            metric_cols.append("risk")
                        if "risk_value" in last_rows.columns:
                            metric_cols.append("risk_value")

                        for c in [
                            "management_fee",
                            "withholding_tax_rate",
                            "market_share",
                            "fund_management_fee",
                            "fund_market_share",
                        ]:
                            if c in last_rows.columns and c not in metric_cols:
                                metric_cols.append(c)

                        metrics_table = []
                        for col in metric_cols:
                            row = {"Metric": pretty(col)}
                            for code in selected_codes:
                                val = last_rows[last_rows["code"] == code][col]
                                if val.empty:
                                    row[code] = "-"
                                else:
                                    v = val.iloc[0]
                                    if col in ["market_cap", "number_of_investors"]:
                                        row[code] = fmt_int(v)
                                    else:
                                        row[code] = fmt_percent(v)
                            metrics_table.append(row)

                        if metrics_table:
                            mt_df = pd.DataFrame(metrics_table)
                            st.markdown("### Last Day Metrics")
                            st.dataframe(mt_df, hide_index=True)
                        else:
                            st.info("No metric columns available for selected funds.")

                        st.markdown("---")
                        st.markdown(
                            "### Return Analysis (1W, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y)"
                        )

                        def period_return(df_code, start_date, end_date):
                            sub = df_code[
                                (df_code["date"] >= start_date)
                                & (df_code["date"] <= end_date)
                            ].sort_values("date")
                            if len(sub) < 2:
                                return np.nan
                            p0 = sub["price"].iloc[0]
                            p1 = sub["price"].iloc[-1]
                            if p0 <= 0:
                                return np.nan
                            return (p1 / p0 - 1.0) * 100

                        periods = [
                            (_t("1 Hafta", "1 Week"), 7),
                            (_t("1 Ay", "1 Month"), 30),
                            (_t("3 Ay", "3 Months"), 90),
                            (_t("6 Ay", "6 Months"), 180),
                            (_t("1 Yıl", "1 Year"), 365),
                            (_t("3 Yıl", "3 Years"), 365 * 3),
                            (_t("5 Yıl", "5 Years"), 365 * 5),
                        ]

                        rows = []
                        for code in selected_codes:
                            d_code = comp[comp["code"] == code].copy()
                            if d_code.empty:
                                continue

                            d_code = d_code.sort_values("date")
                            end_date_code = d_code["date"].max()

                            ytd_start = datetime(end_date_code.year, 1, 1)

                            row_vals = {"Fund": code}

                            for lbl, days in periods:
                                start_date = end_date_code - timedelta(days=days)
                                r = period_return(d_code, start_date, end_date_code)
                                row_vals[lbl] = (
                                    fmt_percent(r) if not np.isnan(r) else "-"
                                )

                            r_ytd = period_return(d_code, ytd_start, end_date_code)
                            row_vals["Year-To-Date"] = (
                                fmt_percent(r_ytd) if not np.isnan(r_ytd) else "-"
                            )

                            rows.append(row_vals)

                        if rows:
                            ret_df = pd.DataFrame(rows)
                            st.dataframe(ret_df, hide_index=True)
                        else:
                            st.info("Not enough price data to calculate returns.")

            # -------- STATISTICS PAGE --------
            elif main_page == "stats":
                st.subheader("📌 Fund Statistics – Performance, Risk & Distribution")
                st.markdown("### Fund-level Summary (Performance & Risk)")

                if data.empty or "price" not in data.columns:
                    st.warning(
                        "⚠️ Column **'price'** is missing or no data. "
                        "Return and volatility statistics cannot be computed for this selection.",
                    )
                else:
                    stats_list = []
                    data_sorted = data.copy()
                    data_sorted["date"] = pd.to_datetime(data_sorted["date"])
                    data_sorted = data_sorted.sort_values(["code", "date"])
                    data_sorted["price"] = pd.to_numeric(
                        data_sorted["price"], errors="coerce"
                    )

                    for code, dfg in data_sorted.groupby("code"):
                        dfg = dfg.dropna(subset=["price"])
                        if len(dfg) < 2:
                            continue

                        prices = dfg["price"]
                        first_price = prices.iloc[0]
                        last_price = prices.iloc[-1]

                        total_return = (last_price / first_price - 1.0) * 100
                        daily_ret = prices.pct_change().dropna()

                        if daily_ret.empty:
                            ann_vol = None
                            hit_ratio = None
                            best_day = None
                            worst_day = None
                            sortino = None
                        else:
                            ann_vol = daily_ret.std() * (252 ** 0.5) * 100
                            positive_days = (daily_ret > 0).sum()
                            hit_ratio = positive_days / len(daily_ret) * 100
                            best_day = daily_ret.max() * 100
                            worst_day = daily_ret.min() * 100

                            downside_ret = daily_ret[daily_ret < 0]
                            if downside_ret.empty or downside_ret.std() == 0:
                                sortino = None
                            else:
                                downside_std = downside_ret.std()
                                sortino = (
                                    daily_ret.mean() / downside_std
                                ) * (252 ** 0.5)

                        cum_max = prices.cummax()
                        drawdown = prices / cum_max - 1
                        max_dd = drawdown.min() * 100

                        n_days = (dfg["date"].iloc[-1] - dfg["date"].iloc[0]).days
                        if n_days > 0:
                            ann_return = (last_price / first_price) ** (
                                252 / n_days
                            ) - 1
                        else:
                            ann_return = None

                        if ann_return is not None and max_dd < 0:
                            calmar = ann_return / abs(max_dd / 100)
                        else:
                            calmar = None

                        if "market_cap" in dfg.columns:
                            aum = pd.to_numeric(
                                dfg["market_cap"], errors="coerce"
                            ).mean()
                        else:
                            aum = None

                        stats_list.append(
                            {
                                "code": code,
                                "Period Start": dfg["date"].iloc[0].date(),
                                "Period End": dfg["date"].iloc[-1].date(),
                                "Total Return (%)": total_return,
                                "Annual Volatility (%)": ann_vol,
                                "Max Drawdown (%)": max_dd,
                                "Hit Ratio (%)": hit_ratio,
                                "Best Day (%)": best_day,
                                "Worst Day (%)": worst_day,
                                "Avg Market Cap": aum,
                                "Calmar Ratio": calmar,
                                "Sortino Ratio": sortino,
                            }
                        )

                    if not stats_list:
                        st.info(
                            "There is not enough price history in the selected date range "
                            "to compute performance statistics for the chosen funds.",
                        )
                    else:
                        stats_df = pd.DataFrame(stats_list)

                        format_map = {
                            "Total Return (%)": fmt_percent,
                            "Annual Volatility (%)": fmt_percent,
                            "Max Drawdown (%)": fmt_percent,
                            "Hit Ratio (%)": fmt_percent,
                            "Best Day (%)": fmt_percent,
                            "Worst Day (%)": fmt_percent,
                            "Avg Market Cap": fmt_int,
                            "Calmar Ratio": fmt_float,
                            "Sortino Ratio": fmt_float,
                        }

                        stats_df_display = stats_df.rename(
                            columns={
                                "code": "Code",
                                "Period Start": "Start",
                                "Period End": "End",
                                "Total Return (%)": "Total Return",
                                "Annual Volatility (%)": "Ann. Volatility",
                                "Max Drawdown (%)": "Max Drawdown",
                                "Hit Ratio (%)": "Hit Ratio",
                                "Best Day (%)": "Best Day",
                                "Worst Day (%)": "Worst Day",
                                "Avg Market Cap": "Avg Market Cap",
                                "Calmar Ratio": "Calmar",
                                "Sortino Ratio": "Sortino",
                            }
                        )

                        stats_styled = stats_df_display.style.format(format_map)
                        st.dataframe(stats_styled, hide_index=True)

                st.markdown("---")
                st.markdown("### Per-fund Return Distribution, Sharpe & Interpretation")

                if data.empty or "price" not in data.columns:
                    st.warning(
                        "⚠️ Column **'price'** is missing. "
                        "Daily returns, distribution and Sharpe ratio cannot be computed.",
                    )
                else:
                    codes_unique = sorted(data["code"].unique())
                    selected_code_stats = st.selectbox(
                        "Select fund for distribution",
                        codes_unique,
                        key="dist_code",
                    )

                    d_stats = data[data["code"] == selected_code_stats].copy()
                    d_stats["date"] = pd.to_datetime(d_stats["date"])
                    d_stats = d_stats.sort_values("date")
                    d_stats["price"] = pd.to_numeric(
                        d_stats["price"], errors="coerce"
                    )
                    d_stats = d_stats.dropna(subset=["price"])

                    ret = d_stats["price"].pct_change().dropna()

                    if ret.empty:
                        st.info(
                            f"There is not enough price data for **{selected_code_stats}** "
                            f"in the selected period to compute return distribution.",
                        )
                    else:
                        mean_ret = ret.mean() * 100

                        if len(ret) < 2 or ret.std() == 0:
                            ann_vol = None
                            sharpe = None
                        else:
                            ann_vol = ret.std() * (252 ** 0.5) * 100
                            sharpe = (ret.mean() / ret.std()) * (252 ** 0.5)

                        downside_ret_pf = ret[ret < 0]
                        if downside_ret_pf.empty or downside_ret_pf.std() == 0:
                            sortino_pf = None
                        else:
                            downside_std_pf = downside_ret_pf.std()
                            sortino_pf = (
                                ret.mean() / downside_std_pf
                            ) * (252 ** 0.5)

                        positive_days = (ret > 0).sum()
                        hit_ratio_pf = positive_days / len(ret) * 100

                        best_day_pf = ret.max() * 100
                        worst_day_pf = ret.min() * 100

                        cum_max_pf = d_stats["price"].cummax()
                        drawdown_pf = d_stats["price"] / cum_max_pf - 1
                        max_dd_pf = drawdown_pf.min() * 100

                        n_days_pf = (
                            d_stats["date"].iloc[-1] - d_stats["date"].iloc[0]
                        ).days
                        if n_days_pf > 0:
                            ann_return_pf = (
                                d_stats["price"].iloc[-1] / d_stats["price"].iloc[0]
                            ) ** (252 / n_days_pf) - 1
                        else:
                            ann_return_pf = None

                        if ann_return_pf is not None and max_dd_pf < 0:
                            calmar_pf = ann_return_pf / abs(max_dd_pf / 100)
                        else:
                            calmar_pf = None

                        col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(
                            [1, 1, 1, 1, 1.6, 1, 1]
                        )
                        col_a.metric("Mean Daily Return", f"{mean_ret:.2f}%")

                        if ann_vol is not None:
                            col_b.metric("Annual Volatility", f"{ann_vol:.2f}%")
                        else:
                            col_b.metric("Annual Volatility", "Insufficient data")

                        if sharpe is not None:
                            col_c.metric("Sharpe (rf ≈ 0)", f"{sharpe:.2f}")
                        else:
                            col_c.metric("Sharpe (rf ≈ 0)", "Not reliable")

                        col_d.metric("Hit Ratio", f"{hit_ratio_pf:.2f}%")
                        col_e.metric(
                            "Best / Worst Day",
                            f"{best_day_pf:.2f}% / {worst_day_pf:.2f}%",
                        )

                        if calmar_pf is not None:
                            col_f.metric("Calmar", f"{calmar_pf:.2f}")
                        else:
                            col_f.metric("Calmar", "NA")

                        if sortino_pf is not None:
                            col_g.metric("Sortino", f"{sortino_pf:.2f}")
                        else:
                            col_g.metric("Sortino", "NA")

                        fig_hist = px.histogram(
                            ret * 100,
                            nbins=30,
                            labels={"value": "Daily Return (%)"},
                            title=f"{selected_code_stats} – Daily Return Distribution",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.markdown("#### Automatic Interpretation")

                        if mean_ret > 0.05:
                            trend_text = (
                                f"- **Trend:** The fund exhibits a clear positive bias with an "
                                f"average daily return of **{mean_ret:.2f}%**."
                            )
                        elif mean_ret > 0:
                            trend_text = (
                                f"- **Trend:** The fund has a mildly positive drift with an "
                                f"average daily return of **{mean_ret:.2f}%**."
                            )
                        elif mean_ret < -0.05:
                            trend_text = (
                                f"- **Trend:** The fund shows a negative bias with an "
                                f"average daily return of **{mean_ret:.2f}%**."
                            )
                        else:
                            trend_text = (
                                f"- **Trend:** The fund is broadly flat over the period with an "
                                f"average daily return of **{mean_ret:.2f}%**."
                            )

                        if ann_vol is None:
                            vol_text = (
                                "- **Volatility:** There is not enough data to estimate "
                                "a robust annualized volatility figure."
                            )
                        else:
                            if ann_vol < 5:
                                vol_bucket_en = "very low"
                            elif ann_vol < 10:
                                vol_bucket_en = "low"
                            elif ann_vol < 20:
                                vol_bucket_en = "moderate"
                            else:
                                vol_bucket_en = "high"

                            vol_text = (
                                f"- **Volatility:** Annualized volatility of "
                                f"**{ann_vol:.2f}%** can be classified as **{vol_bucket_en}**."
                            )

                        if hit_ratio_pf >= 60:
                            hit_text = (
                                f"- **Consistency:** A hit ratio of **{hit_ratio_pf:.2f}%** "
                                f"indicates that the fund finishes most days in positive territory."
                            )
                        elif hit_ratio_pf >= 40:
                            hit_text = (
                                f"- **Consistency:** A hit ratio of **{hit_ratio_pf:.2f}%** "
                                f"suggests a fairly balanced mix of up and down days."
                            )
                        else:
                            hit_text = (
                                f"- **Consistency:** A hit ratio of **{hit_ratio_pf:.2f}%** "
                                f"means the fund has more negative than positive days."
                            )

                        risk_text = (
                            f"- **Risk Events:** The best single day was **{best_day_pf:.2f}%**, "
                            f"the worst single day was **{worst_day_pf:.2f}%**, and the maximum "
                            f"drawdown over the period was around **{max_dd_pf:.2f}%**."
                        )

                        if sharpe is None:
                            sharpe_text = (
                                "- **Risk-adjusted Return (Sharpe):** The Sharpe ratio cannot be "
                                "reliably computed (insufficient history or near-zero volatility)."
                            )
                        else:
                            if sharpe > 1.5:
                                q_en = "strong"
                            elif sharpe > 1.0:
                                q_en = "solid"
                            elif sharpe > 0.5:
                                q_en = "acceptable"
                            else:
                                q_en = "weak"

                            sharpe_text = (
                                f"- **Risk-adjusted Return (Sharpe):** A Sharpe ratio of "
                                f"**{sharpe:.2f}** points to **{q_en}** "
                                "risk-adjusted performance over the selected period."
                            )

                        if calmar_pf is None:
                            calmar_text = (
                                "- **Capital Efficiency (Calmar):** Calmar ratio cannot be "
                                "computed reliably (no drawdown or insufficient history)."
                            )
                        else:
                            if calmar_pf > 2:
                                q_en = "very strong"
                            elif calmar_pf > 1:
                                q_en = "strong"
                            elif calmar_pf > 0.5:
                                q_en = "moderate"
                            else:
                                q_en = "weak"

                            calmar_text = (
                                f"- **Capital Efficiency (Calmar):** A Calmar ratio of "
                                f"**{calmar_pf:.2f}** suggests **{q_en}** "
                                "return per unit of maximum drawdown."
                            )

                        if sortino_pf is None:
                            sortino_text = (
                                "- **Downside Risk (Sortino):** Sortino ratio cannot be computed "
                                "reliably (no downside volatility observed)."
                            )
                        else:
                            if sortino_pf > 2:
                                q_en = "very strong"
                            elif sortino_pf > 1:
                                q_en = "strong"
                            elif sortino_pf > 0.5:
                                q_en = "acceptable"
                            else:
                                q_en = "weak"

                            sortino_text = (
                                f"- **Downside Risk (Sortino):** A Sortino ratio of "
                                f"**{sortino_pf:.2f}** indicates **{q_en}** "
                                "return per unit of downside volatility."
                            )

                        st.markdown(
                            "\n".join(
                                [
                                    trend_text,
                                    vol_text,
                                    hit_text,
                                    risk_text,
                                    sharpe_text,
                                    calmar_text,
                                    sortino_text,
                                ]
                            )
                        )

            # -------- PROBABILITY PAGE --------
            elif main_page == "prob":
                st.subheader(
                    "📌 Probability Analysis – Monthly Up/Down & 12-month Scenarios",
                )

                if data.empty or "price" not in data.columns:
                    st.warning(
                        "⚠️ Column **'price'** is missing. "
                        "A monthly up/down probability model cannot be built for this selection.",
                    )
                else:
                    codes_unique = sorted(data["code"].unique())
                    selected_code_prob = st.selectbox(
                        "Select fund for probability analysis",
                        codes_unique,
                        key="prob_code",
                    )

                    d_prob = data[data["code"] == selected_code_prob].copy()
                    d_prob["date"] = pd.to_datetime(d_prob["date"])
                    d_prob = d_prob.sort_values("date")
                    d_prob["price"] = pd.to_numeric(
                        d_prob["price"], errors="coerce"
                    )
                    d_prob = d_prob.dropna(subset=["price"])

                    if d_prob.empty:
                        st.info(
                            f"There is not enough price history for **{selected_code_prob}** "
                            f"to run the probability model in the chosen period.",
                        )
                    else:
                        monthly_price = (
                            d_prob.set_index("date")["price"]
                            .resample("M")
                            .last()
                            .dropna()
                        )

                        monthly_ret = monthly_price.pct_change().dropna()
                        if monthly_ret.empty:
                            st.info(
                                "Monthly returns cannot be computed from the available data. "
                                "Try extending the date range for a longer history.",
                            )
                        else:
                            n_m = len(monthly_ret)
                            n_up_m = (monthly_ret > 0).sum()
                            n_down_m = (monthly_ret < 0).sum()
                            n_flat_m = (monthly_ret == 0).sum()

                            p_up_m = n_up_m / n_m

                            se_p = math.sqrt(p_up_m * (1 - p_up_m) / n_m)
                            z95 = 1.96
                            ci_low_m = max(0.0, p_up_m - z95 * se_p)
                            ci_up_m = min(1.0, p_up_m + z95 * se_p)

                            rng = np.random.default_rng(12345)
                            B = 5000
                            arr_m = (monthly_ret > 0).astype(int).values
                            boot_ps_m = rng.choice(
                                arr_m, size=(B, len(arr_m)), replace=True
                            ).mean(axis=1)
                            boot_low_m, boot_up_m = np.percentile(
                                boot_ps_m, [2.5, 97.5]
                            )

                            st.markdown("#### Monthly history")

                            col1p, col2p, col3p, col4p = st.columns(4)
                            col1p.metric("Total Months", n_m)
                            col2p.metric("Up Months", n_up_m)
                            col3p.metric("Down Months", n_down_m)
                            col4p.metric("Flat Months", n_flat_m)

                            st.markdown("")
                            st.markdown("#### Up-month probability")

                            col5p, col6p, col7p = st.columns(3)
                            col5p.metric("Empirical p(up)", f"{p_up_m*100:.1f}%")
                            col6p.metric(
                                "95% CI (normal)",
                                f"[{ci_low_m*100:.1f}%, {ci_up_m*100:.1f}%]",
                            )
                            col7p.metric(
                                "95% CI (bootstrap)",
                                f"[{boot_low_m*100:.1f}%, {boot_up_m*100:.1f}%]",
                            )

                            st.markdown("#### Interpretation")
                            st.markdown(
                                f"""
- Based on **{n_m}** monthly observations, the probability that **{selected_code_prob}** closes a month *up* is about **{p_up_m*100:.1f}%**.
- Using a **normal approximation**, the 95% confidence interval for this probability is roughly **{ci_low_m*100:.1f}% – {ci_up_m*100:.1f}%**.
- Using **bootstrap resampling**, the 95% confidence interval is roughly **{boot_low_m*100:.1f}% – {boot_up_m*100:.1f}%**.
"""
                            )

                            st.markdown("---")
                            st.markdown(
                                "### 12-month Forward Scenarios (Binomial Model)",
                            )

                            months_forward = 12
                            k_vals_12 = np.arange(0, months_forward + 1)
                            pmf_12 = [
                                binom_pmf(k, months_forward, p_up_m)
                                for k in k_vals_12
                            ]

                            E_X_12 = months_forward * p_up_m
                            prob_ge_6 = 1 - binom_cdf(5, months_forward, p_up_m)
                            prob_ge_8 = 1 - binom_cdf(7, months_forward, p_up_m)
                            prob_ge_9 = 1 - binom_cdf(8, months_forward, p_up_m)
                            prob_le_5 = binom_cdf(5, months_forward, p_up_m)

                            st.markdown("#### Scenario metrics (next 12 months)")

                            c1, c2, c3 = st.columns(3)
                            c1.metric(
                                "Expected # of positive months (12)",
                                f"{E_X_12:.2f}",
                            )
                            c2.metric(
                                "P(X ≥ 6 positive months)",
                                f"{prob_ge_6*100:.1f}%",
                            )
                            c3.metric(
                                "P(X ≥ 8 positive months)",
                                f"{prob_ge_8*100:.1f}%",
                            )

                            c4, c5 = st.columns(2)
                            c4.metric(
                                "P(X ≥ 9 positive months)",
                                f"{prob_ge_9*100:.1f}%",
                            )
                            c5.metric(
                                "P(X ≤ 5 positive months)",
                                f"{prob_le_5*100:.1f}%",
                            )

                            fig_binom = px.bar(
                                x=k_vals_12,
                                y=pmf_12,
                                labels={
                                    "x": "Number of positive months in 12",
                                    "y": "Probability",
                                },
                                title=f"{selected_code_prob} – Binomial Distribution for Next 12 Months",
                            )
                            st.plotly_chart(fig_binom, use_container_width=True)

                            summary_lines_en = [
                                "**Scenario Summary (12 months):**",
                                "",
                                f"- If the future behaves similarly to the past, on average "
                                f"about **{E_X_12:.1f}** of the next 12 months are expected "
                                f"to close **up**.",
                                f"- The probability of having **at least 6** positive months "
                                f"is about **{prob_ge_6*100:.0f}%**.",
                                f"- A stronger “bullish” scenario with **at least 8** positive "
                                f"months has probability around **{prob_ge_8*100:.0f}%**, and a "
                                f"very strong year with **≥ 9** up months has probability "
                                f"roughly **{prob_ge_9*100:.0f}%**.",
                                f"- On the downside, the chance that **5 or fewer** months end "
                                f"positive (i.e. a negative-dominant year) is about "
                                f"**{prob_le_5*100:.0f}%**.",
                            ]

                            st.markdown("\n".join(summary_lines_en))

            # -------- ARBITRAGE PAGE (BIST50, yfinance, TRY ONLY) --------
            elif main_page == "arb":
                st.subheader("🔁 Arbitrage / Relative Value – BIST 50 (Prices in TRY)")

                # 1) YFinance'dan BIST 50 verisini çek (TRY)
                eq = fetch_bist50_prices_yf(effective_start, effective_end)
                if eq.empty:
                    st.error(
                        "❌ BIST 50 equities data could not be fetched from yfinance for the selected date range."
                    )
                    st.stop()

                # TRY fiyatı kullan
                eq["price"] = eq["price_try"]
                data_eq = eq[["date", "code", "price"]].copy()
                data_eq["date"] = pd.to_datetime(data_eq["date"])
                data_eq["price"] = pd.to_numeric(data_eq["price"], errors="coerce")
                data_eq = data_eq.dropna(subset=["price"])

                all_universe = sorted(data_eq["code"].unique())

                st.info("Universe: BIST 50 (Yahoo Finance .IS, prices in TRY).")

                # ---- KULLANICININ GİRDİĞİ HİSSELER: ÜSTTEKİ 'Fund Codes' INPUTUNDAN ----
                tickers_raw = codes_text.strip()

                if not tickers_raw:
                    st.info(
                        "Please enter at least one BIST 50 ticker in the 'Fund Codes (comma separated)' box above (e.g. AEFES or AEFES, AKBNK)."
                    )
                    st.stop()

                requested = [
                    c.strip().upper() for c in tickers_raw.split(",") if c.strip()
                ]

                selected_universe = [c for c in requested if c in all_universe]
                invalid_universe = [c for c in requested if c not in all_universe]

                if invalid_universe:
                    st.warning(
                        "These codes are not in the BIST 50 universe and will be ignored: "
                        + ", ".join(invalid_universe)
                    )

                if not selected_universe:
                    st.error(
                        "No valid equities found in the BIST 50 universe. "
                        "Please check that the codes you enter are BIST 50 constituents (e.g. AEFES, AKBNK, THYAO)."
                    )
                    st.stop()

                st.markdown(
                    f"**Active universe (from your input & BIST 50):** {', '.join(selected_universe)}"
                )

                # ---------------------------
                # SINGLE STOCK MEAN REVERSION
                # ---------------------------
                st.markdown("## 📉 Single Stock Mean Reversion")

                single_code = st.selectbox(
                    "Choose equity:",
                    selected_universe,
                    key="arb_single_equity",
                )

                d = data_eq[data_eq["code"] == single_code].sort_values("date")
                d = d.dropna(subset=["price"])

                if len(d) < 30:
                    st.warning(
                        "At least 30 daily observations are required for mean-reversion analysis."
                    )
                else:
                    win = 20
                    d["ma"] = d["price"].rolling(win).mean()
                    d["std"] = d["price"].rolling(win).std()
                    d["z"] = (d["price"] - d["ma"]) / d["std"]

                    last = d.dropna().iloc[-1]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Last Price (TRY)", f"{last['price']:.2f}")
                    c2.metric(f"{win}-day Avg", f"{last['ma']:.2f}")
                    c3.metric("Z-score", f"{last['z']:.2f}")

                    fig = px.line(
                        d,
                        x="date",
                        y=["price", "ma"],
                        title=f"{single_code} – Price vs {win}-day MA (TRY)",
                    )
                    fig.update_layout(
                        legend_title_text="Series",
                        xaxis_title="Date",
                        yaxis_title="Price (TRY)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Günlük % değişim tablosu (GUN.F%)
                    st.markdown("### 📋 Daily % Change (GUN.F%)")
                    d_table = d[["date", "price"]].copy().sort_values("date")
                    d_table["daily_change_pct"] = d_table["price"].pct_change() * 100
                    d_table["date"] = d_table["date"].dt.strftime("%Y-%m-%d")

                    d_table_display = d_table.rename(
                        columns={
                            "date": "Date",
                            "price": "Close (TRY)",
                            "daily_change_pct": "Daily % Change",
                        }
                    )
                    d_table_display["Daily % Change"] = d_table_display[
                        "Daily % Change"
                    ].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

                    st.dataframe(d_table_display, hide_index=True)

                    # Yorum
                    z_val = last["z"]
                    if z_val >= 2:
                        st.success(
                            "📈 Z ≥ 2 → Stock is **expensive vs its recent mean** → potential SELL/SHORT zone (after risk checks)."
                        )
                    elif z_val <= -2:
                        st.success(
                            "📉 Z ≤ -2 → Stock is **cheap vs its recent mean** → potential BUY/LONG zone (after risk checks)."
                        )
                    elif abs(z_val) < 1:
                        st.info(
                            "Price is close to its recent average (|Z| < 1) → no strong overbought/oversold signal."
                        )
                    else:
                        st.info(
                            "Moderate deviation (1 ≤ |Z| < 2) → signal is weak, better to monitor than trade."
                        )

                # ---------------------------
                # PAIRS ARBITRAGE
                # ---------------------------
                st.markdown("---")

                if len(selected_universe) < 2:
                    st.info(
                        "To enable pair arbitrage, enter at least two BIST 50 equities in the 'Fund Codes' box (e.g. AEFES, AKBNK)."
                    )
                else:
                    st.markdown(
                        "## 🔀 Pairs Arbitrage (Relative Value Between Two Equities)"
                    )

                    b1, b2 = st.columns(2)
                    base = b1.selectbox(
                        "Base equity (long leg)",
                        selected_universe,
                        key="arb_base_bist",
                    )
                    hedge_candidates = [x for x in selected_universe if x != base]
                    hedge = b2.selectbox(
                        "Hedge equity (short leg)",
                        hedge_candidates,
                        key="arb_hedge_bist",
                    )

                    pivot = data_eq.pivot_table(
                        index="date", columns="code", values="price"
                    ).dropna()

                    if base not in pivot.columns or hedge not in pivot.columns:
                        st.warning(
                            "No overlapping price history for the selected pair of equities."
                        )
                    else:
                        p = pivot[[base, hedge]].dropna()
                        if len(p) < 60:
                            st.warning(
                                "At least 60 overlapping observations are required for pair arbitrage analysis."
                            )
                        else:
                            spread = np.log(p[base] / p[hedge])
                            win_spread = 60
                            m = spread.rolling(win_spread).mean()
                            s = spread.rolling(win_spread).std()
                            z_spread = (spread - m) / s

                            df_spread = pd.DataFrame(
                                {
                                    "date": spread.index,
                                    "spread": spread.values,
                                    "mean": m.values,
                                    "+1σ": (m + s).values,
                                    "-1σ": (m - s).values,
                                }
                            ).dropna()

                            fig2 = px.line(
                                df_spread,
                                x="date",
                                y=["spread", "mean", "+1σ", "-1σ"],
                                title=(
                                    f"{base}/{hedge} – Log spread & bands "
                                    f"(rolling {win_spread}-day, TRY)"
                                ),
                            )
                            fig2.update_layout(
                                xaxis_title="Date",
                                yaxis_title="log(P_base / P_hedge)",
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                            z_last = z_spread.dropna().iloc[-1]
                            st.metric("Spread Z-score", f"{z_last:.2f}")

                            if z_last >= 2:
                                st.success(
                                    f"📈 Spread Z ≈ {z_last:.2f} → **{base} expensive vs {hedge}** → classic setup: SHORT {base} / LONG {hedge} (after risk checks)."
                                )
                            elif z_last <= -2:
                                st.success(
                                    f"📉 Spread Z ≈ {z_last:.2f} → **{base} cheap vs {hedge}** → classic setup: LONG {base} / SHORT {hedge}."
                                )
                            else:
                                st.info(
                                    f"Spread Z ≈ {z_last:.2f} → spread is inside normal band (|Z| < 2); no clear arbitrage signal."
                                )

                            st.markdown(
                                """
> ⚠️ This is a **purely statistical** analysis. Before any real trade, always check:
> - Liquidity & bid–ask cost  
> - Short-selling ability and borrow cost  
> - Taxation & regulations  
> - Your own risk limits / stop-loss rules  
"""
                            )

