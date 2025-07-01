import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from openai import OpenAI
import textwrap
from typing import Optional
import calendar
from dateutil import parser
from dateparser.search import search_dates

app = FastAPI()

# Load Data from GitHub
CSV_URL = "https://raw.githubusercontent.com/SwapnilGautama/Power-BI-AI-Chatbot/main/operational_data_full_jan_to_mar_2025.csv"

df = pd.read_csv(CSV_URL, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])
df["WIP Days"] = (df["End Date"] - df["Start Date"]).dt.days

# KPI summary preprocessing (example: Monthly WIP summary)
df["Month"] = pd.to_datetime(df["Start Date"]).dt.to_period("M").astype(str)
monthly_kpi_summary = (
    df.groupby("Month")
    .agg({
        "WIP Days": "mean",
        "Start Date": "count",
        "Pend Case": lambda x: (x.astype(str).str.lower() == "yes").sum()
    })
    .rename(columns={"Start Date": "Total Cases", "Pend Case": "Total Pends"})
    .round(1)
    .to_dict("index")
)

# Data model for input
class ChatRequest(BaseModel):
    question: str
    month: Optional[str] = None
    year: Optional[int] = 2025

# WIP analysis logic (simplified example)
def generate_prescriptive_response(month_name=None, year=2025):
    if not month_name:
        month = pd.to_datetime(df["Start Date"].max()).month
    else:
        try:
            month = list(calendar.month_name).index(month_name.capitalize())
        except ValueError:
            return f"Invalid month '{month_name}'."

    month_str = f"{year}-{month:02d}"
    raw_monthly = df[df["Month"] == month_str]

    if raw_monthly.empty:
        return f"No KPI data found for {calendar.month_name[month]} {year}."

    top_sources = raw_monthly["Source"].value_counts().head(3).to_dict()

    return {
        "summary": f"WIP analysis for {calendar.month_name[month]} {year}",
        "top_sources": top_sources,
        "total_cases": int(raw_monthly.shape[0]),
        "avg_wip_days": round(raw_monthly["WIP Days"].mean(), 1)
    }

# Chat endpoint
@app.post("/chat")
async def chat_query(request: ChatRequest):
    try:
        question = request.question

        if "wip" in question.lower() and any(m in question.lower() for m in calendar.month_name if m):
            tokens = question.lower().split()
            month_token = next((word for word in tokens if word.capitalize() in calendar.month_name), None)
            return generate_prescriptive_response(month_token, request.year)

        else:
            # General fallback using OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = textwrap.dedent(f"""
            You are an AI assistant for operations analytics. Answer the following question using real data:
            ---
            Data summary:
            {df.describe(include='all').fillna('-').to_string()[:1000]}...

            Question: {question}
            """)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful analytics assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            return {"response": response.choices[0].message.content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Example test: run with uvicorn like:
# uvicorn powerbi_chatbot_backend:app --reload

# Run with: uvicorn main:app --host=0.0.0.0 --port=8000
