import streamlit as st
import pandas as pd
from tefas import Crawler
from datetime import datetime
import plotly.express as px


def fetch_tefas(codes, start, end, kind):
    crawler = Crawler()
    frames = []
    for code in codes:
        df = crawler.fetch(start=start, end=end, name=code, kind=kind)
        if df is not None and not df.empty:
            df["code"] = code
            frames.append(df)
        else:
            st.warning(f"⚠️ No data found for: {code}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("date")


# ------------------------------------------
# NUMBER FORMATTING HELPERS
# ------------------------------------------

def fmt_int(x):
    """Format like 2.411.844.935"""
    try:
        s = f"{int(float(x)):,}"
        return s.replace(",", ".")
    except:
        return x

def fmt_percent(x):
    """7.28 -> 7.28%"""
    try:
        return f"{float(x):.2f}%"
    except:
        return x

def fmt_float(x):
    """8.640000 -> 8.64"""
    try:
        return f"{float(x):.2f}"
    except:
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
# STREAMLIT DASHBOARD
# ------------------------------------------

st.set_page_config(page_title="TEFAS Dashboard", page_icon="📈", layout="wide")
st.title("📈 TEFAS Fund Dashboard")

codes_text = st.text_input("Fund Codes (comma separated):", "SPN")
kind = st.selectbox("Fund Type", ["YAT", "EMK", "BYF"])

col1, col2 = st.columns(2)
start = col1.date_input("Start Date", datetime(2025, 10, 19))
end = col2.date_input("End Date", datetime(2025, 11, 18))


if st.button("📥 Fetch Data"):
    codes = [c.strip().upper() for c in codes_text.split(",") if c.strip()]
    # st.info("⏳ Fetching data...")

    data = fetch_tefas(codes, str(start), str(end), kind)

    if data.empty:
        st.error("❌ No data returned.")
    else:
        # st.success("✅ Data fetched successfully!")

        # Date formatting (remove time part)
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        # Move all-zero columns to the far right
        data = move_zero_columns_last(data)

        # ---------------------
        # COLUMN GROUPS
        # ---------------------
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

        # All other numeric columns (e.g. price, tmm, etc.)
        float_cols = [
            c for c in data.columns
            if c not in int_cols + percent_cols
            and pd.api.types.is_numeric_dtype(data[c])
        ]

        # For delta calculation we will use all numeric columns:
        delta_cols = int_cols + percent_cols + float_cols

        # ==========================
        # TABS (ORDER: RAW → DELTAS → CHARTS)
        # ==========================
        tab_raw, tab_deltas, tab_charts = st.tabs(
            ["Raw Data", "Deltas", "Charts"]
        )

        # -------- Raw Data Tab (FIRST) --------
        with tab_raw:
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

        # -------- Deltas Tab (MIDDLE) --------
        with tab_deltas:
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

        # -------- Charts Tab (LAST) --------
        with tab_charts:
            st.subheader("📊 Price Chart")

            # 1) Price chart (all codes)
            fig_price = px.line(data, x="date", y="price", color="code")
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Asset Allocation – Latest Day")

            if percent_cols:
                codes_unique = sorted(data["code"].unique())
                selected_code = st.selectbox(
                    "Select code for allocation charts", codes_unique
                )

                latest_alloc = (
                    data[data["code"] == selected_code]
                    .sort_values("date")
                    .tail(1)
                )

                alloc_df = latest_alloc[percent_cols].T.reset_index()
                alloc_df.columns = ["Category", "Value"]
                alloc_df["Category"] = alloc_df["Category"].apply(pretty)

                # Sıralama: büyükten küçüğe
                alloc_df = alloc_df.sort_values("Value", ascending=False)

                # 2) Pie chart
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

                # 3) Treemap – daha net, tam genişlik
                st.caption("Treemap (sorted by weight)")
                fig_tree = px.treemap(
                    alloc_df,
                    path=["Category"],
                    values="Value",
                    title=f"Asset Allocation Treemap - {selected_code}",
                )
                # Kutuların üstüne label + yüzde yaz
                fig_tree.update_traces(
                    texttemplate="%{label}<br>%{value:.2f}%",
                    hovertemplate="<b>%{label}</b><br>Share: %{value:.2f}%<extra></extra>",
                )
                fig_tree.update_layout(height=600)
                st.plotly_chart(fig_tree, use_container_width=True)

            else:
                st.info("No allocation (percentage) columns available for allocation charts.")
