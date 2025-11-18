import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Sellers App", layout="wide")

# Load the data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] == 1:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')

    # Auto detects columns
    cols_map = {c.strip().lower(): c for c in df.columns}

    def find_col(keywords):
        for k, orig in cols_map.items():
            if any(kw in k for kw in keywords):
                return orig
        return None

    region_col = find_col(["region"])
    sold_col   = find_col(["sold", "units"])
    total_col  = find_col(["total", "sales"])
    avg_col    = find_col(["average", "avg"])
    name_col   = find_col(["name"])
    last_col   = find_col(["last"])

    # Create Vendor column if not present
    if "Vendor" not in df.columns:
        if name_col and last_col:
            df["Vendor"] = (df[name_col].astype(str) + " " + df[last_col].astype(str)).str.strip()
        elif name_col:
            df["Vendor"] = df[name_col].astype(str).str.strip()
        else:
            df["Vendor"] = "Unknown Vendor"

    # Make numeric to avoid mean/sum errors
    for col in [sold_col, total_col] + ([avg_col] if avg_col else []):
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure required columns exist
    if not region_col or not sold_col or not total_col:
        st.error("Could not detect required columns (Region, Units Sold, Total Sales). Check your CSV headers.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")
    regions = sorted(df[region_col].dropna().unique().tolist())
    chosen_regions = st.sidebar.multiselect("Region", regions, placeholder="Select region(s)")

    # Derived filtered data
    df_f = df[df[region_col].isin(chosen_regions)] if chosen_regions else df.copy()

    # Optional filters reset button
    if st.sidebar.button("Reset filters"):
        chosen_regions = []
        df_f = df.copy()

    # Layout: table + chart + vendor detail
    st.title("Sellers Dashboard")

    # 1) Display the table, filtered by Region
    with st.container():
        st.subheader("Table (filtered by Region)")
        st.dataframe(df_f, width='stretch', hide_index=True)

    # 2) Graphs of Units Sold, Total Sales, and Average Sales (by Region)
    with st.container():
        st.subheader("Graphs")
        by_region = df_f.groupby(region_col, as_index=False).agg(
            **{
                "Units Sold": (sold_col, "sum"),
                "Total Sales": (total_col, "sum"),
                "Average Sales": (avg_col, "mean") if (avg_col and avg_col in df_f.columns) else (total_col, "sum"),
            }
        )

        # If no explicit average column, compute at group level
        if not (avg_col and avg_col in df_f.columns):
            by_region["Average Sales"] = by_region["Total Sales"] / by_region["Units Sold"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Units Sold by Region")
            fig1 = px.bar(by_region, x=region_col, y="Units Sold")
            st.plotly_chart(fig1, width='stretch')
        with c2:
            st.caption("Total Sales by Region")
            fig2 = px.bar(by_region, x=region_col, y="Total Sales")
            st.plotly_chart(fig2, width='stretch')
        with c3:
            st.caption("Average Sales by Region")
            fig3 = px.bar(by_region, x=region_col, y="Average Sales")
            st.plotly_chart(fig3, width='stretch')

    # 3) Display data for a specific vendor (from the filtered set)
    with st.container():
        st.subheader("Vendor Detail")
        vendors = sorted(df_f["Vendor"].unique().tolist())
        selected_vendor = st.selectbox("Choose a vendor", ["— Select —"] + vendors, index=0)

        if selected_vendor != "— Select —":
            vdf = df_f[df_f["Vendor"] == selected_vendor]
            k1, k2, k3 = st.columns(3)
            k1.metric("Units Sold (sum)", f"{pd.to_numeric(vdf[sold_col], errors='coerce').sum():,.0f}")
            k2.metric("Total Sales (sum)", f"{pd.to_numeric(vdf[total_col], errors='coerce').sum():,.0f}")

            if avg_col and avg_col in vdf.columns:
                avg_value = pd.to_numeric(vdf[avg_col], errors='coerce').mean()
            else:
                total_sum = pd.to_numeric(vdf[total_col], errors='coerce').sum()
                units_sum = pd.to_numeric(vdf[sold_col], errors='coerce').sum()
                avg_value = (total_sum / units_sum) if units_sum else float("nan")

            k3.metric("Sales Average (mean)", f"{avg_value:,.3f}")
            st.dataframe(vdf, width='stretch', hide_index=True)