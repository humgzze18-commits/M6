import pandas as pd
import numpy as np
import streamlit as st
import os, json, joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_ml_artifacts(path="artifacts"):
    price_model = joblib.load(os.path.join(path, "xgb_price_fastshallow.joblib"))
    qty_model   = joblib.load(os.path.join(path, "xgb_qty_fastshallow.joblib"))
    encoders    = joblib.load(os.path.join(path, "encoders.joblib"))

    with open(os.path.join(path, "feature_lists.json"), "r") as f:
        feats = json.load(f)

    FEATURES_PRICE = feats["FEATURES_PRICE"]
    FEATURES_QTY   = feats["FEATURES_QTY"]

    work = pd.read_csv(os.path.join(path, "work_engineered.csv"))
    if "DATE" in work.columns:
        work["DATE"] = pd.to_datetime(work["DATE"], errors="coerce")

    return price_model, qty_model, encoders, work, FEATURES_PRICE, FEATURES_QTY

def get_base_row_for_scenario(work_eng, encoders, sku_code, tp_code):
    temp = work_eng.copy()

    # Encode using same encoders as training
    if "SKU" in encoders:
        sku_enc = encoders["SKU"].transform([str(sku_code)])[0]
    else:
        sku_enc = sku_code

    if "TP_GROUP" in encoders:
        tp_enc = encoders["TP_GROUP"].transform([str(tp_code)])[0]
    else:
        tp_enc = tp_code

    subset = temp[(temp["SKU"] == sku_enc) & (temp["TP_GROUP"] == tp_enc)]

    if subset.empty:
        subset = temp[temp["SKU"] == sku_enc]
    if subset.empty:
        subset = temp[temp["TP_GROUP"] == tp_enc]
    if subset.empty:
        subset = temp

    subset = subset.sort_values("DATE")
    return subset.iloc[-1].copy()


st.set_page_config(page_title="Whirlpool Console (Prototype)", layout="wide")

CSV_PATH = "base_consolidada_modified.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Dates
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["YEAR"] = df["DATE"].dt.year
        df["MONTH_TS"] = df["DATE"].dt.to_period("M").dt.to_timestamp()
    # Numerics (coerce safely)
    for c in [
        "QTY","INV","EXCHANGE_RATE","PRICE_LIST_DA","PRICE_FINAL_DA",
        "VPC","WTY","VAR_FW","VAR_SGA","FEM","POLICY"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Simple revenue proxy
    if {"PRICE_FINAL_DA","QTY"}.issubset(df.columns):
        df["REVENUE"] = df["PRICE_FINAL_DA"] * df["QTY"]
    else:
        df["REVENUE"] = np.nan
    return df

df = load_data(CSV_PATH)

st.title("Whirlpool – Historical Insights & Prediction Console (Prototype)")

# --- Quick dataset checks (collapsible)
with st.expander("Dataset snapshot & schema", expanded=False):
    st.dataframe(df.head(15))
    st.code(df.dtypes.astype(str).to_string())
    if "DATE" in df.columns and not df["DATE"].isna().all():
        st.write("Date range:", df["DATE"].min(), "→", df["DATE"].max())
    if "TP_GROUP" in df.columns:
        st.write("Partners (#):", df["TP_GROUP"].nunique())
    if "SKU" in df.columns:
        st.write("SKUs (#):", df["SKU"].nunique())

# Initialize non-widget state
if "sel_tps" not in st.session_state:
    st.session_state.sel_tps = []
if "sel_skus" not in st.session_state:
    st.session_state.sel_skus = []
if "sku_entry_val" not in st.session_state:
    st.session_state.sku_entry_val = ""
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    if "DATE" in df.columns and not df["DATE"].isna().all():
        st.session_state.start_date = df["DATE"].min().date()
        st.session_state.end_date   = df["DATE"].max().date()
    else:
        st.session_state.start_date = None
        st.session_state.end_date   = None


# Sidebar filters (widgets)
with st.sidebar:
    page = st.radio(
        "Page",
        ["Historical Overview", "Predictions & Scenarios"],
        index=0
    )

with st.sidebar:
    st.header("Filters")

    # Partners
    tp_opts = sorted(df["TP_GROUP"].dropna().unique().tolist()) if "TP_GROUP" in df.columns else []
    _sel_tps = st.multiselect(
        "Trade Partner(s)",
        options=tp_opts,
        default=st.session_state.sel_tps,
        key="sel_tps_widget"
    )
    # copy widget value into our store key
    st.session_state.sel_tps = _sel_tps

    # SKU input: type to add (default shows ALL until at least one added)
    df_tp = df[df["TP_GROUP"].isin(st.session_state.sel_tps)] if st.session_state.sel_tps else df
    sku_pool = set(df_tp["SKU"].dropna().unique().tolist()) if "SKU" in df.columns else set()

    st.markdown("**Add SKU by code**")
    new_sku = st.text_input(
        "Type an exact SKU code and click 'Add'",
        value=st.session_state.sku_entry_val,
        key="sku_entry_widget",
        placeholder="e.g., 8MWTW2024WJM"
    )

    c_add, c_clear = st.columns([1,1])
    if c_add.button("Add SKU"):
        if new_sku:
            if new_sku in sku_pool:
                if new_sku not in st.session_state.sel_skus:
                    st.session_state.sel_skus.append(new_sku)
                    st.success(f"Added SKU: {new_sku}")
                else:
                    st.info("That SKU is already selected.")
            else:
                st.warning("No records for that SKU under the current Trade Partner filter.")
            # clear local backing value and rerun
            st.session_state.sku_entry_val = ""
            st.rerun()

    if c_clear.button("Clear all SKUs", type="secondary"):
        st.session_state.sel_skus = []
        st.session_state.sku_entry_val = ""
        st.rerun()

    # Show and allow removal of selected SKUs
    if st.session_state.sel_skus:
        chosen = st.multiselect(
            "Selected SKUs (deselect to remove)",
            options=st.session_state.sel_skus,
            default=st.session_state.sel_skus,
            key="sel_skus_widget"
        )
        # sync back to store
        st.session_state.sel_skus = chosen
    else:
        st.caption("No SKUs selected → showing all SKUs.")

    # Dates (two single pickers, clamped; widget keys differ from store keys)
    if "DATE" in df.columns and not df["DATE"].isna().all():
        min_d, max_d = df["DATE"].min().date(), df["DATE"].max().date()
        _sd = st.date_input(
            "Start date",
            value=st.session_state.start_date,
            min_value=min_d, max_value=max_d,
            key="start_date_widget"
        )
        _ed = st.date_input(
            "End date",
            value=st.session_state.end_date,
            min_value=min_d, max_value=max_d,
            key="end_date_widget"
        )
        # enforce order and clamp
        _sd = max(min_d, min(_sd, max_d))
        _ed = max(_sd,   min(_ed, max_d))
        st.session_state.start_date, st.session_state.end_date = _sd, _ed
    else:
        st.session_state.start_date = st.session_state.end_date = None


# Apply filters
df_f = df.copy()

# Partners filter
if st.session_state.sel_tps:
    df_f = df_f[df_f["TP_GROUP"].isin(st.session_state.sel_tps)]

# SKU filter (only if user has added any)
if st.session_state.sel_skus:
    df_f = df_f[df_f["SKU"].isin(st.session_state.sel_skus)]

# Date filter
if st.session_state.start_date and st.session_state.end_date and "DATE" in df_f.columns:
    start = pd.to_datetime(st.session_state.start_date)
    end   = pd.to_datetime(st.session_state.end_date)
    df_f = df_f[(df_f["DATE"] >= start) & (df_f["DATE"] <= end)]

# Friendly message + KPIs only on Historical page
if page == "Historical Overview":
    if df_f.empty:
        # Specific hint if the SKUs exist globally but not for the chosen TP
        if st.session_state.sel_skus:
            missing_for_tp = [s for s in st.session_state.sel_skus if s not in df_tp["SKU"].values]
            if missing_for_tp:
                st.info(
                    "No records for the selected combination. "
                    "These SKUs have no rows for the current Trade Partner filter: "
                    + ", ".join(missing_for_tp)
                )
            else:
                st.info("No records for the selected Trade Partner(s), SKU(s), and date range.")
    else:
        # KPI row
        def kpis(d: pd.DataFrame):
            total_qty = int(d["QTY"].sum()) if "QTY" in d.columns else 0
            avg_price = float(d["PRICE_FINAL_DA"].mean()) if "PRICE_FINAL_DA" in d.columns else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric("Units", f"{total_qty:,}")
            c2.metric("Avg Final Price", "—" if np.isnan(avg_price) else "${:,.2f}".format(avg_price))
            if "INV" in d.columns:
                c3.metric("Avg Inventory", "{:,.0f}".format(d["INV"].mean()))

        st.subheader("KPIs")
        kpis(df_f)


import altair as alt
from datetime import timedelta
# Period columns for charts
if page == "Historical Overview":
    if "DATE" in df.columns and not df["DATE"].isna().all():
        # Month (already computed as MONTH_TS in load_data)
        # Quarter
        if "QUARTER_TS" not in df.columns:
            df["QUARTER_TS"] = df["DATE"].dt.to_period("Q").dt.to_timestamp()

        # Week (start of ISO week)
        if "WEEK_TS" not in df.columns:
            # ISO week start (Monday). Streamlit/Altair are fine with datetime.
            df["WEEK_TS"] = df["DATE"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    def pick_skus_for_charts(d: pd.DataFrame, user_skus: list, top_n: int = 5) -> list:
        """
        If user_skus is non-empty, return them (intersection with data).
        Else return Top-N SKUs ranked by sum(REVENUE) within d.
        """
        if "SKU" not in d.columns:
            return []

        if user_skus:
            pool = set(d["SKU"].dropna().unique().tolist())
            return [s for s in user_skus if s in pool]

        # rank by sum(REVENUE) only for picking, not for display
        if {"REVENUE","SKU"}.issubset(d.columns) and not d.empty:
            top = (
                d.groupby("SKU", as_index=False)["REVENUE"].sum()
                .sort_values("REVENUE", ascending=False)
                .head(top_n)["SKU"]
                .tolist()
            )
            return top
        return []

    # OVERVIEW (charts below KPIs)
    st.markdown("### Overview")

    if df_f.empty:
        st.info("No records under current filters to display charts.")
    else:
        # Graph-only controls (do NOT affect KPIs)
        gcol1, gcol2, gcol3 = st.columns([1,1,2])

        with gcol1:
            granularity = st.selectbox(
                "Granularity",
                options=["Month", "Quarter", "Week"],
                index=0,
                key="granularity_overview"
            )

        with gcol2:
            top_n = st.selectbox(
                "Top-N SKUs",
                options=[3,5,10],
                index=1,  # default 5
                key="topn_overview"
            )

        # Week-only 12-week window
        week_start, week_end = None, None
        if granularity == "Week" and "DATE" in df_f.columns and not df_f.empty:
            min_d, max_d = df_f["DATE"].min().date(), df_f["DATE"].max().date()
            with gcol3:
                st.caption("Limit weekly view to a ≤12-week window.")
                _ws = st.date_input("Start (≤12 weeks window)", value=min_d, min_value=min_d, max_value=max_d, key="week_start_overview")
                # propose end = start + 83 days (~12 weeks)
                proposed_end = min(max_d, _ws + timedelta(days=83))
                _we = st.date_input("End (≤12 weeks window)", value=proposed_end, min_value=_ws, max_value=min(max_d, _ws + timedelta(days=83)), key="week_end_overview")

                # clamp to <= 83 days
                if (_we - _ws).days > 83:
                    _we = _ws + timedelta(days=83)
                    st.warning("Weekly range reduced to 12 weeks max.")
                week_start, week_end = _ws, _we

        # Prepare the dataframe for charts (does not touch df_f used by KPIs)
        d_chart = df_f.copy()

        # When week granularity is on, restrict to the 12-week window (if provided)
        if granularity == "Week" and week_start and week_end:
            d_chart = d_chart[(d_chart["DATE"] >= pd.to_datetime(week_start)) &
                            (d_chart["DATE"] <= pd.to_datetime(week_end))]

        # Pick SKUs for charts only (Top-N by revenue if none selected)
        chart_skus = pick_skus_for_charts(d_chart, st.session_state.get("sel_skus", []), top_n=top_n)
        if chart_skus:
            d_chart = d_chart[d_chart["SKU"].isin(chart_skus)]

        # If after picking SKUs we have nothing, show friendly message
        if d_chart.empty:
            st.info("No chartable data after applying chart-only controls (granularity / Top-N / 12-week window).")
        else:
            left, right = st.columns([2, 1])

            # Left column: time series
            with left:
                # Choose x-axis & group by based on granularity
                if granularity == "Month":
                    if "MONTH_TS" in d_chart.columns:
                        tkey = "MONTH_TS"
                    else:
                        tkey = "DATE"
                    res = (d_chart.groupby([tkey, "SKU"], as_index=False)
                                .agg(QTY=("QTY", "sum"), PRICE=("PRICE_FINAL_DA","mean")))
                elif granularity == "Quarter":
                    tkey = "QUARTER_TS" if "QUARTER_TS" in d_chart.columns else "DATE"
                    res = (d_chart.groupby([tkey, "SKU"], as_index=False)
                                .agg(QTY=("QTY", "sum"), PRICE=("PRICE_FINAL_DA","mean")))
                else:  # Week
                    tkey = "WEEK_TS" if "WEEK_TS" in d_chart.columns else "DATE"
                    res = (d_chart.groupby([tkey, "SKU"], as_index=False)
                                .agg(QTY=("QTY", "sum"), PRICE=("PRICE_FINAL_DA","mean")))

                st.subheader("Price over time (by SKU)")
                price_chart = (
                    alt.Chart(res)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{tkey}:T", title=granularity),
                        y=alt.Y("PRICE:Q", title="Avg Final Price"),
                        color=alt.Color("SKU:N", title="SKU"),
                        tooltip=[alt.Tooltip(f"{tkey}:T", title=granularity),
                                "SKU:N",
                                alt.Tooltip("PRICE:Q", title="Avg Price", format=",.2f"),
                                alt.Tooltip("QTY:Q", title="Qty", format=",")]
                    )
                    .properties(height=260)
                    .interactive()
                )
                st.altair_chart(price_chart, use_container_width=True)

                st.subheader("Quantity over time (by SKU)")
                qty_chart = (
                    alt.Chart(res)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{tkey}:T", title=granularity),
                        y=alt.Y("QTY:Q", title="Quantity"),
                        color=alt.Color("SKU:N", title="SKU"),
                        tooltip=[alt.Tooltip(f"{tkey}:T", title=granularity),
                                "SKU:N",
                                alt.Tooltip("QTY:Q", title="Qty", format=","),
                                alt.Tooltip("PRICE:Q", title="Avg Price", format=",.2f")]
                    )
                    .properties(height=260)
                    .interactive()
                )
                st.altair_chart(qty_chart, use_container_width=True)

            # Right column: partner snapshots (qty / avg price)
            with right:
                st.subheader("Top Partners by Quantity")
                tp_qty = (d_chart.groupby("TP_GROUP", as_index=False)["QTY"].sum()
                                .sort_values("QTY", ascending=False)
                                .head(10))
                if not tp_qty.empty:
                    tp_qty_chart = (
                        alt.Chart(tp_qty)
                        .mark_bar()
                        .encode(
                            x=alt.X("QTY:Q", title="Quantity"),
                            y=alt.Y("TP_GROUP:N", sort="-x", title="Trade Partner"),
                            tooltip=["TP_GROUP:N", alt.Tooltip("QTY:Q", format=",")]
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(tp_qty_chart, use_container_width=True)
                else:
                    st.caption("No partner data for quantity.")

                st.subheader("Top Partners by Avg Price")
                tp_price = (d_chart.groupby("TP_GROUP", as_index=False)["PRICE_FINAL_DA"].mean()
                                    .rename(columns={"PRICE_FINAL_DA":"AVG_PRICE"})
                                    .sort_values("AVG_PRICE", ascending=False)
                                    .head(10))
                if not tp_price.empty:
                    tp_price_chart = (
                        alt.Chart(tp_price)
                        .mark_bar()
                        .encode(
                            x=alt.X("AVG_PRICE:Q", title="Avg Final Price"),
                            y=alt.Y("TP_GROUP:N", sort="-x", title="Trade Partner"),
                            tooltip=["TP_GROUP:N", alt.Tooltip("AVG_PRICE:Q", format=",.2f")]
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(tp_price_chart, use_container_width=True)
                else:
                    st.caption("No partner data for price.")

# SECOND PAGE – Predictions & ML
if page == "Predictions & Scenarios":

    st.title("Predictions & Scenario Builder (ML)")
    st.write(
        "This page uses the **same trained XGBoost FastShallow models** and encoders "
        "you exported from Colab. Final price and weekly quantity are both predicted."
    )

    with st.spinner("Loading ML models…"):
        price_model, qty_model, encoders, work_eng, FEATURES_PRICE, FEATURES_QTY = load_ml_artifacts()

    # --- 1) Select Trade Partner (dropdown) ---
    all_tp = sorted(df["TP_GROUP"].dropna().unique().tolist()) if "TP_GROUP" in df.columns else []
    tp_sel = st.selectbox("Trade Partner", all_tp, key="ml_tp_sel")

    # Limit SKUs universe to this TP (for validation), but input is manual
    df_tp_subset = df[df["TP_GROUP"] == tp_sel] if tp_sel and "TP_GROUP" in df.columns else df
    available_skus_tp = set(df_tp_subset["SKU"].dropna().unique().tolist()) if "SKU" in df.columns else set()
    available_skus_all = set(df["SKU"].dropna().unique().tolist()) if "SKU" in df.columns else set()

    # --- 2) SKU typed by user (must exist in the database) ---
    sku_input = st.text_input(
        "SKU code (must exist in the database)",
        value="",
        key="ml_sku_input",
        placeholder="e.g., 8MWTW2024WJM"
    ).strip()

    valid_sku = False
    sku_message_placeholder = st.empty()

    if sku_input:
        if sku_input in available_skus_all:
            valid_sku = True
            if sku_input not in available_skus_tp:
                sku_message_placeholder.warning(
                    "This SKU exists in the database, but there are no records for the selected Trade Partner. "
                    "The model will fall back to using global history for that SKU."
                )
            else:
                sku_message_placeholder.success("Valid SKU for this Trade Partner.")
        else:
            sku_message_placeholder.error("This SKU does not exist in the database.")
            valid_sku = False

    if not sku_input or not valid_sku:
        st.info("Enter a valid SKU code to configure and run a scenario.")
        st.stop()  # don't render controls if invalid

    # --- 3) Base row from engineered ML dataset (same logic as notebook) ---
    base_row = get_base_row_for_scenario(work_eng, encoders, sku_input, tp_sel)

    st.markdown("### Scenario Controls")

    c1, c2 = st.columns(2)

    # ========== Column 1: Discount (required) ==========
    with c1:
        default_disc = float(base_row.get("DISCOUNT_PCT", 0.0))
        disc_in = st.slider(
            "Discount % (required)",
            min_value=0.0,
            max_value=0.8,
            value=float(np.clip(default_disc, 0.0, 0.8)),
            step=0.01
        )
        st.caption("This represents the promotional intensity for the scenario week.")

    # ========== Column 2: Week of year (required) ==========
    with c2:
        week_default = int(base_row.get("WEEK", 26)) if not pd.isna(base_row.get("WEEK", np.nan)) else 26
        week_in = st.slider(
            "Week of Year (required)",
            min_value=1,
            max_value=52,
            value=week_default,
            step=1
        )

        # Map week to a future date in 2026 (prediction year)
        scenario_date = pd.Timestamp("2026-01-05") + pd.to_timedelta(week_in - 1, unit="W")  # Mondays
        approx_month_name = scenario_date.strftime("%B")
        month_in = scenario_date.month

        st.caption(f"Scenario prediction date: **Week {week_in}, {approx_month_name} 2026**")

    # --- 4) Build scenario row for prediction (only discount + week/month changed) ---
    def build_scenario_row(base_r, encs):
        s = base_r.copy()

        # Encode SKU and TP_GROUP exactly as in training
        if "SKU" in encs:
            s["SKU"] = encs["SKU"].transform([str(sku_input)])[0]
        else:
            s["SKU"] = sku_input

        if "TP_GROUP" in encs:
            s["TP_GROUP"] = encs["TP_GROUP"].transform([str(tp_sel)])[0]
        else:
            s["TP_GROUP"] = tp_sel

        # Discount (mandatory override)
        if "DISCOUNT_PCT" in s.index:
            s["DISCOUNT_PCT"] = disc_in

        # Week / Month / sin/cos seasonal features
        s["MONTH"] = month_in
        s["WEEK"] = week_in
        s["sin_woy"] = np.sin(2 * np.pi * week_in / 52.0)
        s["cos_woy"] = np.cos(2 * np.pi * week_in / 52.0)

        return s

    scenario_row = build_scenario_row(base_row, encoders)
    X_row = scenario_row.to_frame().T

    # Ensure all required features exist (fill with 0.0 if missing)
    for col in FEATURES_PRICE:
        if col not in X_row.columns:
            X_row[col] = 0.0
    for col in FEATURES_QTY:
        if col not in X_row.columns:
            X_row[col] = 0.0

    # Force numeric dtypes for all model features to avoid XGBoost dtype errors
    feature_union = list(dict.fromkeys(FEATURES_PRICE + FEATURES_QTY))
    for col in feature_union:
        if col in X_row.columns:
            X_row[col] = pd.to_numeric(X_row[col], errors="coerce")

    # --- 5) Predict final price and weekly quantity ---
    if st.button("Predict final price & weekly quantity", type="primary"):
        with st.spinner("Predicting with trained XGBoost models…"):
            # PRICE model: predict log of final price
            log_pred_price = float(price_model.predict(X_row[FEATURES_PRICE])[0])
            pred_price = float(np.expm1(log_pred_price))

            # QTY model: predict log of quantity
            log_pred_qty = float(qty_model.predict(X_row[FEATURES_QTY])[0])
            pred_qty = float(np.expm1(log_pred_qty))

        st.markdown("### Scenario Prediction")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Final Price", f"{pred_price:,.2f}")
        c2.metric("Predicted Weekly Quantity", f"{pred_qty:,.0f} units")

        st.caption(
            f"Scenario: TP = **{tp_sel}**, SKU = **{sku_input}**, Week = **{week_in} ({approx_month_name} 2026)**, "
            f"Discount = **{disc_in:.0%}**."
        )

        # ========================
        # 6) Historical vs Scenario Charts
        # ========================
        st.markdown("### Historical Context vs Prediction")

        # Filter historical data for this TP × SKU
        hist = df[(df["TP_GROUP"] == tp_sel) & (df["SKU"] == sku_input)].copy()
        if "DATE" in hist.columns:
            hist = hist.sort_values("DATE")

            # Define 1.5 years (≈ 78 weeks) window ending at scenario_date
            window_start = scenario_date - pd.Timedelta(weeks=78)

            # We only have history up to the last available date (e.g., mid-2025)
            # so this will naturally show a gap between last DATE and the 2026 scenario point.
            hist_window = hist[hist["DATE"] >= window_start].copy()

            # Aggregate weekly if there are multiple records per DATE
            # (if data is already weekly per Monday, this is basically a no-op)
            hist_weekly = (
                hist_window.groupby("DATE", as_index=False)
                .agg(
                    QTY=("QTY", "sum"),
                    PRICE_FINAL_DA=("PRICE_FINAL_DA", "mean")
                )
            )
            hist_weekly["TYPE"] = "Historical"

            # Scenario point as a future "DATE"
            pred_point = pd.DataFrame({
                "DATE": [scenario_date],
                "QTY": [pred_qty],
                "PRICE_FINAL_DA": [pred_price],
                "TYPE": ["Prediction"]
            })

            # Combine for plotting
            price_data = pd.concat([hist_weekly[["DATE", "PRICE_FINAL_DA", "TYPE"]], pred_point[["DATE", "PRICE_FINAL_DA", "TYPE"]]], ignore_index=True)
            qty_data   = pd.concat([hist_weekly[["DATE", "QTY", "TYPE"]], pred_point[["DATE", "QTY", "TYPE"]]], ignore_index=True)

            # Price chart
            st.subheader("Final Price – Historical vs Scenario")
            base_price = alt.Chart(price_data).encode(
                x=alt.X("DATE:T", title="Date")
            )

            price_hist_line = base_price.transform_filter(
                alt.datum.TYPE == "Historical"
            ).mark_line(point=True).encode(
                y=alt.Y("PRICE_FINAL_DA:Q", title="Final Price"),
                color=alt.value("#1f77b4"),
                tooltip=[
                    alt.Tooltip("DATE:T", title="Date"),
                    alt.Tooltip("PRICE_FINAL_DA:Q", title="Price", format=",.2f")
                ]
            )

            price_pred_point = base_price.transform_filter(
                alt.datum.TYPE == "Prediction"
            ).mark_point(size=120, filled=True).encode(
                y="PRICE_FINAL_DA:Q",
                color=alt.value("#ff7f0e"),
                tooltip=[
                    alt.Tooltip("DATE:T", title="Scenario Date"),
                    alt.Tooltip("PRICE_FINAL_DA:Q", title="Predicted Price", format=",.2f")
                ]
            )

            st.altair_chart(price_hist_line + price_pred_point, use_container_width=True)

            # Quantity chart
            st.subheader("Quantity – Historical vs Scenario")
            base_qty = alt.Chart(qty_data).encode(
                x=alt.X("DATE:T", title="Date")
            )

            qty_hist_line = base_qty.transform_filter(
                alt.datum.TYPE == "Historical"
            ).mark_line(point=True).encode(
                y=alt.Y("QTY:Q", title="Quantity"),
                color=alt.value("#1f77b4"),
                tooltip=[
                    alt.Tooltip("DATE:T", title="Date"),
                    alt.Tooltip("QTY:Q", title="Qty", format=",")
                ]
            )

            qty_pred_point = base_qty.transform_filter(
                alt.datum.TYPE == "Prediction"
            ).mark_point(size=120, filled=True).encode(
                y="QTY:Q",
                color=alt.value("#ff7f0e"),
                tooltip=[
                    alt.Tooltip("DATE:T", title="Scenario Date"),
                    alt.Tooltip("QTY:Q", title="Predicted Qty", format=",")
                ]
            )

            st.altair_chart(qty_hist_line + qty_pred_point, use_container_width=True)

            st.caption(
                "The lines show the historical behavior of this TP × SKU for the last ~1.5 years. "
                "The orange dot is the **future scenario prediction in 2026**."
            )
        else:
            st.info("No DATE column available to build the historical vs prediction charts.")