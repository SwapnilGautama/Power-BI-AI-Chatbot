# main.py 

from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import dateparser
from dateparser.search import search_dates
import calendar
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- APP SETUP --------------------
app = FastAPI(title="Power BI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DATA LOAD --------------------
csv_url = "https://raw.githubusercontent.com/SwapnilGautama/ChatBot-Test/refs/heads/main/operational_data_full_jan_to_mar_2025.csv"
df = pd.read_csv(csv_url, dayfirst=True, parse_dates=["Start Date", "End Date", "Target Date"])
df["WIP Days"] = (df["End Date"] - df["Start Date"]).dt.days
df["WIP Days"] = df["WIP Days"].fillna((pd.Timestamp.now() - df["Start Date"]).dt.days).astype(int)

# -------------------- HELPER FUNCTIONS --------------------
def generate_wip_trend_insights(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Start Date"])
    df["Week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # 90-day window
    last_day = df["Date"].max()
    first_day = last_day - pd.Timedelta(days=90)
    trend_df = df[(df["Date"] >= first_day) & (df["Date"] <= last_day)]

    daily_wip = trend_df.groupby("Date").apply(lambda x: x[x["End Date"].isna()].shape[0])
    weekly_wip = trend_df.groupby("Week").apply(lambda x: x[x["End Date"].isna()].shape[0])
    monthly_wip = trend_df.groupby("Month").apply(lambda x: x[x["End Date"].isna()].shape[0])

    pend_df = trend_df[trend_df["Pend Case"].astype(str).str.lower() == "yes"]
    pend_rate = round(pend_df.shape[0] / trend_df.shape[0] * 100, 1) if trend_df.shape[0] > 0 else 0

    def fmt_table(data):
        return "\n".join([f"- **{k}**: {v}" for k, v in data.items()])

    top_sources = trend_df["Source"].value_counts().head(3).to_dict()
    top_portfolios = trend_df["Portfolio"].value_counts().head(3).to_dict()
    top_manual = trend_df["Manual/RPA"].value_counts().head(3).to_dict()
    pend_reasons = pend_df["Pend Reason"].value_counts().head(3).to_dict()

    return f"""
📊 **WIP Trend Insights (Last 3 Months)**

### 🗓️ Monthly WIP:  
{monthly_wip.to_string()}

### 📅 Weekly WIP (Last 12 Weeks):  
{weekly_wip.tail(12).to_string()}

### 📆 Daily WIP (Last 7 Days):  
{daily_wip.tail(7).to_string()}

---

### 🔍 Drivers Behind WIP
- 🔗 **Top Sources**  
{fmt_table(top_sources)}

- 🗂️ **Top Portfolios**  
{fmt_table(top_portfolios)}

- 🤖 **Manual vs RPA**  
{fmt_table(top_manual)}

- 📋 **Pend Rate**: **{pend_rate}%**

- 🧾 **Top Pend Reasons**  
{fmt_table(pend_reasons)}
"""

# -------------------- REQUEST BODY --------------------
class ChatRequest(BaseModel):
    question: str

# -------------------- ROOT ENDPOINT --------------------
@app.get("/")
def root():
    return {"message": "✅ FastAPI is running on Azure successfully!"}

# -------------------- CHAT ENDPOINT --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_question = req.question.strip().lower()

    if "wip" in user_question and "trend" in user_question:
        reply = generate_wip_trend_insights(df)
        return {"response": reply}

    # fallback to GPT if no pattern match
    try:
        summary_text = df.describe(include='all').fillna('-').to_string()
        prompt = f"""
You are Opsi, an expert in operations analytics.

DATA SUMMARY:
{summary_text}

USER QUESTION:
{req.question}

INSTRUCTIONS:
- Use actual metrics (WIP, pend rate, SLA %)
- Avoid vague statements, focus on data
- Bullet points and simple language
- Root causes and patterns if visible
        """

        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful analytics assistant named Opsi."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        reply = response.choices[0].message.content
        return {"response": reply}

    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}
