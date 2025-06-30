import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import textwrap

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import textwrap

# ---------------- JSON SERIALIZATION HELPER ----------------
def safe_json(obj):
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime, date

    def convert(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray, list)):
            return [convert(i) for i in o]
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.strftime('%Y-%m-%d')
        if isinstance(o, pd.Period):
            return str(o)
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}  # <— ✅ convert keys to str
        if isinstance(o, (pd.DataFrame, pd.Series)):
            return convert(o.to_dict())
        return o

    return json.dumps(convert(obj), indent=2, default=str)

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="📊 KPI AI Dashboard", layout="wide")
st.title("📊 BackOffice Operations Dashboard with AI Insights")

# ---------------- LOAD DATA ----------------
import streamlit as st
import pandas as pd

csv_url = "https://raw.githubusercontent.com/SwapnilGautama/ChatBot-Test/refs/heads/main/operational_data_full_jan_to_mar_2025.csv"

try:
    df = pd.read_csv(csv_url, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])
except Exception as e:
    st.error(f"❌ Failed to load CSV from GitHub.\n\n**Error:** `{e}`")
    st.stop()

df["Start Date"] = pd.to_datetime(df["Start Date"], errors='coerce')
df["End Date"] = pd.to_datetime(df["End Date"], errors='coerce')
df["Target Date"] = pd.to_datetime(df["Target Date"], errors='coerce')

# ---------------- FILTERS ----------------
st.sidebar.header("📂 Filters")

start_date, end_date = st.sidebar.date_input(
    "Select Week (Monday to Sunday)",
    [df["Start Date"].min(), df["Start Date"].min() + pd.Timedelta(days=6)]
)
start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
selected_labels = pd.date_range(start=start_date, end=end_date).strftime("%d-%b").tolist()

filtered_df = df[(df["Start Date"] >= start_date) & (df["Start Date"] <= end_date)]

if "Portfolio" in df.columns:
    portfolios = st.sidebar.multiselect("Filter by Portfolio", sorted(df["Portfolio"].dropna().unique()))
    if portfolios:
        filtered_df = filtered_df[filtered_df["Portfolio"].isin(portfolios)]

if "Source" in df.columns:
    sources = st.sidebar.multiselect("Filter by Source", sorted(df["Source"].dropna().unique()))
    if sources:
        filtered_df = filtered_df[filtered_df["Source"].isin(sources)]

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# ---------------- KPI CALCULATION ----------------
df["WIP Days"] = (df["End Date"] - df["Start Date"]).dt.days
df["WIP Days"] = df["WIP Days"].fillna((pd.Timestamp.now() - df["Start Date"]).dt.days).astype(int)
filtered_df["WIP Days"] = df["WIP Days"]

min_date = filtered_df["Start Date"].min()
max_date = max(filtered_df["Start Date"].max(), filtered_df["End Date"].max(), filtered_df["Target Date"].max())
date_range = pd.date_range(start=min_date, end=max_date)

kpi_data = []
pend_rate_values = []
prev_closing_wip = filtered_df[(filtered_df["End Date"].isna()) & (filtered_df["Start Date"] <= min_date)].shape[0]

for report_date in date_range:
    opening_wip = prev_closing_wip
    received_today = filtered_df[filtered_df["Start Date"] == report_date]
    cases_received = received_today.shape[0]
    complete_today = filtered_df[filtered_df["End Date"] == report_date]
    cases_complete = complete_today.shape[0]
    complete_within_sla = complete_today[complete_today["End Date"] < complete_today["Target Date"]].shape[0]
    complete_within_sla_pct = f"{int(round((complete_within_sla / cases_complete * 100)))}%" if cases_complete > 0 else "0%"
    backlog_over_sla = filtered_df[
        (filtered_df["Start Date"] <= report_date) &
        (filtered_df["End Date"].isna()) &
        (filtered_df["Target Date"] < report_date)
    ].shape[0]
    backlog_pct = f"{int(round((backlog_over_sla / prev_closing_wip * 100)))}%" if prev_closing_wip > 0 else "0%"
    wip_in_sla = filtered_df[
        (filtered_df["Start Date"] <= report_date) &
        (filtered_df["End Date"].isna()) &
        (filtered_df["Target Date"] >= report_date)
    ].shape[0]
    closing_wip = opening_wip + cases_received - cases_complete
    wip_in_sla_pct = f"{int(round((wip_in_sla / closing_wip * 100)))}%" if closing_wip > 0 else "0%"

    pend_subset = filtered_df[filtered_df["Start Date"] <= report_date]
    pend_total = pend_subset["Pend Case"].notna().sum()
    pend_yes = pend_subset[pend_subset["Pend Case"].astype(str).str.lower() == "yes"].shape[0]
    pend_rate_val = int(round((pend_yes / pend_total * 100))) if pend_total > 0 else 0
    pend_rate = f"{pend_rate_val}%"

    pend_rate_values.append(pend_rate_val)

    kpi_data.append({
        "Report Date": report_date.strftime("%d-%b"),
        "Opening WIP": opening_wip,
        "Cases Received": cases_received,
        "Cases Complete": cases_complete,
        "Closing WIP": closing_wip,
        "Complete Within SLA": complete_within_sla,
        "Complete Within SLA %": complete_within_sla_pct,
        "Backlog - WIP Over SLA": backlog_over_sla,
        "Backlog %": backlog_pct,
        "WIP in SLA": wip_in_sla,
        "WIP in SLA %": wip_in_sla_pct,
        "Pend Rate": pend_rate
    })

    prev_closing_wip = closing_wip

kpi_df = pd.DataFrame(kpi_data)

# ---------------- ADVANCED KPI ANALYTICS ENGINE ----------------

# 🔹 Trend Frequencies
kpi_df["Report Date Full"] = pd.date_range(start=min_date, end=max_date)

# Daily WIP Trend (already built into KPI)
daily_trend = kpi_df[["Report Date Full", "Closing WIP"]]

# Weekly WIP Trend
kpi_df["Week"] = pd.to_datetime(kpi_df["Report Date Full"]).dt.to_period("W").apply(lambda r: r.start_time)
weekly_trend = kpi_df.groupby("Week")["Closing WIP"].mean().reset_index(name="Avg WIP")

# Monthly WIP Trend
kpi_df["Month"] = pd.to_datetime(kpi_df["Report Date Full"]).dt.to_period("M").astype(str)
monthly_trend = kpi_df.groupby("Month")["Closing WIP"].mean().reset_index(name="Avg WIP")

# Yearly WIP Trend
kpi_df["Year"] = pd.to_datetime(kpi_df["Report Date Full"]).dt.year
yearly_trend = kpi_df.groupby("Year")["Closing WIP"].mean().reset_index(name="Avg WIP")


# 🔹 SLA Compliance & Breach
kpi_df["Complete SLA %"] = kpi_df["Complete Within SLA %"].str.replace("%", "").astype(float)
kpi_df["WIP SLA %"] = kpi_df["WIP in SLA %"].str.replace("%", "").astype(float)

sla_summary = {
    "Avg Complete SLA %": f"{int(kpi_df['Complete SLA %'].mean())}%",
    "Avg WIP SLA %": f"{int(kpi_df['WIP SLA %'].mean())}%"
}


# 🔹 Pend Reasons Distribution (filtered only)
if "Pend Reason" in filtered_df.columns:
    pend_reason_summary = filtered_df["Pend Reason"].value_counts().head(10).to_dict()
else:
    pend_reason_summary = {}


# 🔹 WIP Days Analysis
avg_wip_days = int(df["WIP Days"].mean())
wip_days_trend = df.groupby(df["Start Date"].dt.to_period("W").apply(lambda r: r.start_time))["WIP Days"].mean().reset_index()
wip_days_trend.rename(columns={"WIP Days": "Avg WIP Days"}, inplace=True)

# Outliers in WIP Days
wip_days_q3 = df["WIP Days"].quantile(0.75)
wip_days_outliers = df[df["WIP Days"] > wip_days_q3 + 1.5 * (wip_days_q3 - df["WIP Days"].quantile(0.25))]


# 🔹 Breakdown by Factors
factors = [
    "Portfolio", "Source", "Location", "Event Type", "Process Name",
    "Onshore/Offshore", "Manual/RPA", "Critical", "Vulnerable Customer", "Data Type"
]

breakdown_summary = {}

for factor in factors:
    if factor in df.columns:
        grouped = df.groupby(factor).agg({
            "WIP Days": "mean",
            "Pend Case": lambda x: (x.astype(str).str.lower() == "yes").sum(),
            "Start Date": "count"
        }).rename(columns={"Start Date": "Total Cases"})
        grouped["Avg WIP Days"] = grouped["WIP Days"].round(1)
        breakdown_summary[factor] = grouped[["Avg WIP Days", "Total Cases", "Pend Case"]].sort_values(by="Avg WIP Days", ascending=False)

# 🔹 Monthly KPI Summary for Chatbot Reference
monthly_kpis = df.copy()
monthly_kpis["Month"] = pd.to_datetime(monthly_kpis["Start Date"]).dt.to_period("M").astype(str)
monthly_wip_summary = (
    monthly_kpis.groupby("Month")
    .agg({
        "WIP Days": "mean",
        "Start Date": "count",
        "Pend Case": lambda x: (x.astype(str).str.lower() == "yes").sum()
    })
    .rename(columns={"Start Date": "Total Cases", "Pend Case": "Total Pends"})
    .round(1)
    .to_dict("index")
)

# 🔹 Daily, Weekly, Monthly KPI Summary for Chatbot

# Add parsed day/week/month labels
df["Date"] = pd.to_datetime(df["Start Date"])
df["Week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# Build KPI-level summaries from your existing `kpi_df`
def parse_pct(val):
    return int(val.replace('%', '')) if isinstance(val, str) and '%' in val else 0

kpi_df["Complete SLA % Num"] = kpi_df["Complete Within SLA %"].apply(parse_pct)
kpi_df["WIP SLA % Num"] = kpi_df["WIP in SLA %"].apply(parse_pct)
kpi_df["Pend Rate Num"] = kpi_df["Pend Rate"].apply(parse_pct)

# DAILY
daily_kpi_summary = kpi_df.set_index("Report Date").to_dict("index")

# WEEKLY
kpi_df["Week"] = pd.to_datetime(kpi_df["Report Date Full"]).dt.to_period("W").apply(lambda r: r.start_time)
weekly_kpi_summary = (
    kpi_df.groupby("Week")[["Opening WIP", "Cases Received", "Cases Complete", "Closing WIP",
                            "Complete SLA % Num", "WIP SLA % Num", "Pend Rate Num"]]
    .mean()
    .round(1)
    .to_dict("index")
)

# MONTHLY
kpi_df["Month"] = pd.to_datetime(kpi_df["Report Date Full"]).dt.to_period("M").astype(str)
monthly_kpi_summary = (
    kpi_df.groupby("Month")[["Opening WIP", "Cases Received", "Cases Complete", "Closing WIP",
                             "Complete SLA % Num", "WIP SLA % Num", "Pend Rate Num"]]
    .mean()
    .round(1)
    .to_dict("index")
)
# ---------------- WIP SPIKE ANALYZER FUNCTION ----------------
def analyze_wip_spikes(df_kpi, raw_df):
    df_kpi["Closing WIP Num"] = df_kpi["Closing WIP"]
    rolling_avg = df_kpi["Closing WIP Num"].rolling(window=3).mean()
    df_kpi["WIP Spike"] = df_kpi["Closing WIP Num"] > rolling_avg * 1.2

    spike_days = df_kpi[df_kpi["WIP Spike"] == True]["Report Date"].tolist()
    analysis = []

    for day in spike_days:
        day_raw = raw_df[raw_df["Start Date"].dt.strftime("%d-%b") == day]
        day_kpi = df_kpi[df_kpi["Report Date"] == day]

        pend_total = day_raw["Pend Case"].notna().sum()
        pend_yes = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"].shape[0]
        pend_rate = round((pend_yes / pend_total * 100), 1) if pend_total > 0 else 0

        pend_reason_counts = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"] \
            .groupby("Pend Reason").size().sort_values(ascending=False).to_dict()

        breakdown = {
            "Portfolio": day_raw["Portfolio"].value_counts().head(3).to_dict(),
            "Source": day_raw["Source"].value_counts().head(3).to_dict(),
            "Event Type": day_raw["Event Type"].value_counts().head(3).to_dict(),
            "Manual/RPA": day_raw["Manual/RPA"].value_counts().head(3).to_dict()
        }

        analysis.append({
            "date": day,
            "closing_wip": int(day_kpi["Closing WIP"].values[0]),
            "pend_rate": f"{pend_rate}%",
            "top_pend_reasons": pend_reason_counts,
            "breakdown": breakdown
        })

    return analysis

# ---------------- AI INSIGHTS SECTION ----------------
st.subheader("🧠 AI-Generated Insights")

if st.button("Generate Intelligent Insights with AI"):
    # ⏳ Show spinner while working
    with st.spinner("Analyzing and generating insights..."):

        deep_dive_insights = analyze_wip_spikes(kpi_df, filtered_df)

        story_prompt = f"""
You are a senior operations analyst trusted with providing high-quality, data-backed performance insights.

Below is a filtered performance snapshot, based on the user's selected week and filters. Use this to identify trends, exceptions, patterns, and root causes.

📊 **Performance Summary**:

- **SLA Compliance**
    • Completed within SLA: **{sla_summary['Avg Complete SLA %']}**
    • WIP in SLA: **{sla_summary['Avg WIP SLA %']}**

- **WIP Days**
    • Average WIP Days: **{avg_wip_days}**
    • WIP Outliers: {len(wip_days_outliers)} rows exceeded upper bound

- **WIP Weekly Trend (last 4 weeks)**:
{weekly_trend.tail(4).to_string(index=False)}

- **Monthly Trend**:
{monthly_trend.tail(2).to_string(index=False)}

- **Top 3 Spike Days**:
{"".join([
    f"• {item['date']} – WIP: {item['closing_wip']}, Pend Rate: {item['pend_rate']}, Top Reasons: {', '.join(list(item['top_pend_reasons'].keys())[:2])}\n"
    for item in deep_dive_insights[:3]
])}

- **Top Pend Reasons**:
{safe_json(pend_reason_summary)}

- 📅 **Daily KPI Summary**:
{safe_json(daily_kpi_summary)}

- 📈 **Weekly KPI Summary**:
{safe_json(weekly_kpi_summary)}

- 📆 **Monthly KPI Summary**:
{safe_json(monthly_kpi_summary)}

Now generate **5 strategic insights** that are:
- Actionable and rooted in the data
- 1–2 lines each
- Use actual metrics (%, volumes, comparisons)
- Point out unusual changes or bottlenecks
- Highlight trends vs. previous period if visible
- Use markdown emphasis (bold, bullet points, and emojis like 📈📉✅🛠️)

📌 Format:
- 📌 **[Insight Title]** – supporting metric(s) and explanation.
"""


        try:
            client = OpenAI(api_key=st.secrets["openai_key"])
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in operations analysis."},
                    {"role": "user", "content": story_prompt}
                ],
                temperature=0.5
            )
            gpt_bullets = response.choices[0].message.content
            st.markdown(gpt_bullets)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            
# ---------------- CHARTS ----------------
st.markdown("## 📈 Operational Trends")

chart_df = kpi_df[kpi_df["Report Date"].isin(selected_labels)]
labels = list(chart_df["Report Date"])
cases_received = list(chart_df["Cases Received"])
cases_complete = list(chart_df["Cases Complete"])
closing_wip = list(chart_df["Closing WIP"])
wip_sla_pct = [int(x.replace('%', '')) if '%' in x else 0 for x in chart_df["WIP in SLA %"]]
complete_sla_pct = [int(x.replace('%', '')) if '%' in x else 0 for x in chart_df["Complete Within SLA %"]]
pend_rate_chart = [int(row["Pend Rate"].replace('%', '')) if isinstance(row["Pend Rate"], str) else 0 for _, row in chart_df.iterrows()]

pend_reasons = filtered_df[filtered_df["Start Date"].dt.strftime("%d-%b").isin(labels)]
pend_reason_counts = pend_reasons["Pend Reason"].value_counts().to_dict()
pend_reason_labels = list(pend_reason_counts.keys())
pend_reason_values = list(pend_reason_counts.values())

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📦 Cases Processed vs WIP")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cases Received', x=labels, y=cases_received, marker_color='red'))
    fig.add_trace(go.Bar(name='Cases Complete', x=labels, y=cases_complete, marker_color='blue'))
    fig.add_trace(go.Scatter(name='Closing WIP', x=labels, y=closing_wip, mode='lines+markers', line=dict(color='orange', width=3)))
    fig.update_layout(barmode='group', height=360)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 🧾 SLA Compliance %")
    sla_fig = go.Figure()
    sla_fig.add_trace(go.Scatter(x=labels, y=wip_sla_pct, name='WIP in SLA %', line=dict(color='teal')))
    sla_fig.add_trace(go.Scatter(x=labels, y=complete_sla_pct, name='Complete SLA %', line=dict(color='purple')))
    sla_fig.update_layout(height=360)
    st.plotly_chart(sla_fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🥧 Top Pend Reasons")
    pie_fig = px.pie(
        names=pend_reason_labels,
        values=pend_reason_values,
        title="Pend Reasons Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    pie_fig.update_traces(textinfo='percent+label')
    st.plotly_chart(pie_fig, use_container_width=True)

with col4:
    st.markdown("#### 📊 Pend Rate Trend")
    pend_fig = px.bar(
        x=labels,
        y=pend_rate_chart,
        title="Pend Rate (%)",
        labels={'x': 'Date', 'y': 'Pend Rate (%)'},
        color_discrete_sequence=['orange']
    )
    st.plotly_chart(pend_fig, use_container_width=True)

# ---------------- KPI TABLE ----------------
st.subheader("📋 KPI Table")
st.dataframe(chart_df, use_container_width=True)

# ---------------- WIP SPIKE ANALYZER FUNCTION ----------------
def analyze_wip_spikes(df_kpi, raw_df):
    df_kpi["Closing WIP Num"] = df_kpi["Closing WIP"]
    rolling_avg = df_kpi["Closing WIP Num"].rolling(window=3).mean()
    df_kpi["WIP Spike"] = df_kpi["Closing WIP Num"] > rolling_avg * 1.2

    spike_days = df_kpi[df_kpi["WIP Spike"] == True]["Report Date"].tolist()
    prev_avg_wip = df_kpi["Closing WIP Num"].shift(1).rolling(window=5).mean()

    analysis = []

    for day in spike_days:
        day_raw = raw_df[raw_df["Start Date"].dt.strftime("%d-%b") == day]
        day_kpi = df_kpi[df_kpi["Report Date"] == day]

        pend_total = day_raw["Pend Case"].notna().sum()
        pend_yes = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"].shape[0]
        pend_rate = round((pend_yes / pend_total * 100), 1) if pend_total > 0 else 0

        pend_reason_counts = day_raw[day_raw["Pend Case"].astype(str).str.lower() == "yes"] \
            .groupby("Pend Reason").size().sort_values(ascending=False).to_dict()

        breakdown = {
            "Portfolio": day_raw["Portfolio"].value_counts().head(3).to_dict(),
            "Source": day_raw["Source"].value_counts().head(3).to_dict(),
            "Event Type": day_raw["Event Type"].value_counts().head(3).to_dict(),
            "Manual/RPA": day_raw["Manual/RPA"].value_counts().head(3).to_dict()
        }

        closing_wip = int(day_kpi["Closing WIP"].values[0])
        avg_prev = int(prev_avg_wip[day_kpi.index[0]]) if not pd.isna(prev_avg_wip[day_kpi.index[0]]) else None

        analysis.append({
            "date": day,
            "closing_wip": closing_wip,
            "pend_rate": f"{pend_rate}%",
            "top_pend_reasons": pend_reason_counts,
            "breakdown": breakdown,
            "vs_prev_avg": f"{closing_wip - avg_prev}" if avg_prev else "N/A"
        })

    return analysis

from dateutil import parser
import calendar

def generate_prescriptive_response(kpi_df, raw_df, month_name=None, year=2025):
    if not month_name:
        month = pd.to_datetime(raw_df["Start Date"].max()).month
    else:
        try:
            month = list(calendar.month_name).index(month_name.capitalize())
        except ValueError:
            return f"❌ Couldn't understand the month '{month_name}'. Please provide a valid month."

    month_str = f"{year}-{month:02d}"
    monthly_kpi = kpi_df[kpi_df["Report Date Full"].dt.to_period("M") == pd.Period(month_str)]

    if monthly_kpi.empty:
        return f"⚠️ No KPI data found for {calendar.month_name[month]} {year}."

    raw_monthly = raw_df[raw_df["Start Date"].dt.to_period("M") == pd.Period(month_str)]

    opening_wip = int(monthly_kpi["Opening WIP"].iloc[0])
    closing_wip = int(monthly_kpi["Closing WIP"].iloc[-1])
    avg_closing_wip = int(monthly_kpi["Closing WIP"].mean())
    wip_sla_pct = int(monthly_kpi["WIP SLA % Num"].mean())
    wip_outside_sla_pct = 100 - wip_sla_pct

    top_sources = raw_monthly["Source"].value_counts().head(3).to_dict()
    top_events = raw_monthly["Event Type"].value_counts().head(3).to_dict()
    top_portfolios = raw_monthly["Portfolio"].value_counts().head(3).to_dict()
    high_wip_records = raw_monthly[raw_monthly["WIP Days"] > raw_monthly["WIP Days"].mean() + 2]

    response = f"""
### 📊 WIP Analysis for **{calendar.month_name[month]} {year}**

- **Opening WIP**: {opening_wip}
- **Closing WIP**: {closing_wip}
- **Average WIP (closing)**: {avg_closing_wip}
- ✅ **WIP in SLA**: {wip_sla_pct}%
- ❗ **WIP outside SLA**: {wip_outside_sla_pct}%

---

### 📈 Observations:
- 📉 **WIP changed** from {opening_wip} to {closing_wip}, suggesting {"a reduction" if closing_wip < opening_wip else "an increase"}.
- 🔍 **Top Sources with high WIP**:
  {"".join([f"  • {k}: {v} cases\n" for k, v in top_sources.items()])}
- 🗂️ **Top Event Types**:
  {"".join([f"  • {k}: {v} cases\n" for k, v in top_events.items()])}
- 🏷️ **Top Portfolios**:
  {"".join([f"  • {k}: {v} cases\n" for k, v in top_portfolios.items()])}
- ⚠️ **{high_wip_records.shape[0]} cases** had unusually high WIP days (possible bottlenecks).

---
"""
    return response

def generate_wip_trend_insights(df):
    import numpy as np

    df = df.copy()
    df["Start Date"] = pd.to_datetime(df["Start Date"])
    df["End Date"] = pd.to_datetime(df["End Date"])
    df["Date"] = df["Start Date"]
    df["Week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # 90-day trend window
    last_day = df["Date"].max()
    first_day = last_day - pd.Timedelta(days=90)
    trend_df = df[(df["Date"] >= first_day) & (df["Date"] <= last_day)]

    # Aggregated WIP
    daily_wip = trend_df.groupby("Date").apply(lambda x: x[x["End Date"].isna()].shape[0])
    weekly_wip = trend_df.groupby("Week").apply(lambda x: x[x["End Date"].isna()].shape[0])
    monthly_wip = trend_df.groupby("Month").apply(lambda x: x[x["End Date"].isna()].shape[0])

    last_7 = daily_wip.tail(7)
    last_12w = weekly_wip.tail(12)
    last_3m = monthly_wip.tail(3)

    def horizontal_table(data):
        keys = list(data.keys())
        values = list(data.values())
        header = "| " + " | ".join(keys) + " |"
        separator = "| " + " | ".join(["---"] * len(keys)) + " |"
        row = "| " + " | ".join(str(v) for v in values) + " |"
        return f"{header}\n{separator}\n{row}"

    def format_bullets(d):
        return "\n".join([f"- **{k}**: {v}" for k, v in d.items()])

    # Root causes
    pend_df = trend_df[trend_df["Pend Case"].astype(str).str.lower() == "yes"]
    pend_rate = round(pend_df.shape[0] / trend_df.shape[0] * 100, 1) if trend_df.shape[0] > 0 else 0

    top_sources = trend_df["Source"].value_counts().head(3).to_dict()
    top_portfolios = trend_df["Portfolio"].value_counts().head(3).to_dict()
    top_manual = trend_df["Manual/RPA"].value_counts().head(3).to_dict()
    pend_reasons = pend_df["Pend Reason"].value_counts().head(3).to_dict()

    # 🔍 Detect WIP Increase
    month_keys = list(last_3m.keys())
    callout = ""
    if len(month_keys) >= 2:
        first = last_3m[month_keys[0]]
        second = last_3m[month_keys[1]]
        third = last_3m[month_keys[2]]
        delta1 = second - first
        delta2 = third - second

        if delta1 > 0:
            callout = f"""
🔺 **WIP spiked by {delta1} cases from {month_keys[0]} to {month_keys[1]}**, likely due to:
- 📋 Increase in manual processing: **{top_manual}**
- 🔗 Surge in specific sources: **{top_sources}**
- 🧾 Pend Rate at **{pend_rate}%** with top reasons: {", ".join(pend_reasons.keys())}
"""
        elif delta2 > 0:
            callout = f"""
🔺 **WIP continued to rise from {month_keys[1]} to {month_keys[2]}**, driven by:
- Manual load: **{top_manual}**
- Source mix: **{top_sources}**
- High pend backlog: **{pend_rate}%**
"""

    return f"""
📊 **WIP Trend Insights (Last 3 Months)**

### 🗓️ Monthly WIP  
{horizontal_table(last_3m.to_dict())}

### 📅 Weekly WIP (Last 12 Weeks)  
{horizontal_table({k.strftime('%Y-%m-%d'): v for k, v in last_12w.items()})}

### 📆 Daily WIP (Last 7 Days)  
{horizontal_table({k.strftime('%Y-%m-%d'): v for k, v in last_7.items()})}

---

{callout if callout else '📉 No significant spike detected in WIP trends. WIP is stable or decreasing.'}

---

### 🔍 Drivers Behind WIP
- 🔗 **Top Sources**  
{format_bullets(top_sources)}

- 🗂️ **Top Portfolios**  
{format_bullets(top_portfolios)}

- 🤖 **Manual vs RPA**  
{format_bullets(top_manual)}

- 📋 **Pend Rate**: **{pend_rate}%**

- 🧾 **Top Pend Reasons**  
{format_bullets(pend_reasons)}
"""

# ---------------- AI CHATBOT SECTION ----------------
st.markdown("## 🤖 Meet **Opsi** – Your Analyst Copilot")

# ✅ Load the same CSV used for the dashboard from GitHub
raw_url = "https://raw.githubusercontent.com/SwapnilGautama/AI-Insights-Dashboard/refs/heads/main/operational_data_full_jan_to_mar_2025.csv"

try:
    raw_df = pd.read_csv(raw_url, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])
    raw_df["WIP Days"] = (raw_df["End Date"] - raw_df["Start Date"]).dt.days
    raw_df["WIP Days"] = raw_df["WIP Days"].fillna((pd.Timestamp.now() - raw_df["Start Date"]).dt.days).astype(int)

except Exception as e:
    st.error(f"❌ Failed to load data for chatbot (Opsi).\n\n**Error:** `{e}`")
    st.stop()

# Use your WIP spike analyzer for full data
deep_dive_insights_full = analyze_wip_spikes(kpi_df, raw_df)

# Summarize dataset for GPT prompt
summary_text = f"""
📈 Basic Statistics:
{raw_df.describe(include='all').fillna('-').to_string()}
"""

# Input box
user_question = st.text_input("Ask anything about performance trends:", key="chat_input")

if user_question:
    with st.spinner("Opsi is thinking..."):
        try:
            import dateparser
            from dateparser.search import search_dates
            tokens = user_question.lower().split()
            possible_month = next((word for word in tokens if word.capitalize() in calendar.month_name), None)

            if "wip" in user_question.lower() and "trend" in user_question.lower():
                reply = generate_wip_trend_insights(raw_df)

            elif "wip" in user_question.lower() and "week" in user_question.lower():
                # Clean fuzzy phrasing for better parsing
                cleaned_input = (
                    user_question.lower()
                    .replace("for the week of", "")
                    .replace("the week of", "")
                    .replace("week of", "")
                    .replace("week starting", "")
                    .replace("wip for", "")
                    .replace("wip", "")
                    .replace("th", "")
                    .replace("st", "")
                    .replace("nd", "")
                    .replace("rd", "")
                    .strip()
                )

                parsed_results = search_dates(cleaned_input, settings={"PREFER_DATES_FROM": "past"})
                parsed_date = parsed_results[0][1] if parsed_results else None

                if not parsed_date:
                    # Fallback: "1st week of February", "2nd week of March"
                    import re
                    week_match = re.search(r"(\d+)\s*week of (\w+)", cleaned_input)
                    if week_match:
                        week_num = int(week_match.group(1))
                        month_name = week_match.group(2).capitalize()

                        try:
                            month_dt = pd.to_datetime(f"1 {month_name} 2025")
                            week_start = month_dt + pd.Timedelta(days=(week_num - 1) * 7)
                            week_start = week_start - pd.Timedelta(days=week_start.weekday())
                            week_start_str = week_start.strftime("%Y-%m-%d")
                            reply = generate_weekly_prescriptive_response(kpi_df, raw_df, week_start_str)
                        except Exception:
                            reply = "⚠️ Couldn't parse the week/month combination. Try 'week of 3rd March'."
                    else:
                        reply = "⚠️ I couldn’t understand the week reference. Try something like 'week of 3rd March' or '1st week of Feb'."
                else:
                    week_start = parsed_date - pd.Timedelta(days=parsed_date.weekday())
                    week_start_str = week_start.strftime("%Y-%m-%d")
                    reply = generate_weekly_prescriptive_response(kpi_df, raw_df, week_start_str)

            elif "wip" in user_question.lower() and possible_month:
                reply = generate_prescriptive_response(kpi_df, raw_df, month_name=possible_month)
            else:
                client = OpenAI(api_key=st.secrets["openai_key"])

                prompt = textwrap.dedent(f"""
                You are **Opsi**, an expert in operational analytics and performance reporting.

                You will be given:
                1. A high-level summary of operational data (key statistics and patterns)
                2. A user's analytical question about trends, performance, or root causes.

                Your job:
                - Answer concisely and insightfully using **actual metrics** (e.g. WIP, pend rate, SLA %)
                - Provide **clear explanations**, ideally in **bullet points**
                - Highlight **notable patterns** (spikes, declines, exceptions) and **root causes**
                - Be accurate, data-driven, and use **simple language** for non-technical users
                - Avoid **generic or vague statements**
                - Only output **data-supported insights**
                - Never **repeat the summary directly** — synthesize it into observations

                --- DATA SUMMARY ---
                {summary_text}

                --- USER QUESTION ---
                {user_question}

                Answer:
                """)

                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful analytics assistant named Opsi."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                reply = response.choices[0].message.content

            st.markdown(reply)

        except Exception as e:
            st.error(f"❌ Error: {e}")
