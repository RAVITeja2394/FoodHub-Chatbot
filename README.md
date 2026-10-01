# FoodHub-Chatbot
FoodHub AI chatbot — Streamlit UI + LangGraph/Gemini backend for order status, refunds, and policy support.

# FoodHub ChatBot 🍔

A multi-turn customer support chatbot for FoodHub, built with **Streamlit** (frontend) and **LangGraph + Gemini** (backend). Handles order status, cancellations, refund eligibility/status, payment issues, and policy questions — with SQL-backed order lookups and RAG over the delivery policy document.

## Features
- Customer selection (dropdown or manual Customer ID)
- Multi-turn conversation with session memory (LangGraph `MemorySaver`)
- Intent classification routing: status, policy (RAG), refunds, payment failure, escalation, malicious/out-of-scope blocking
- Quick-question shortcuts
- Auto session end on exit keywords or idle timeout
- Color-coded chat UI with order-status card, feedback icons, transcript download

## Project Structure
```
foodhub-chatbot/
├── frontend/
│   └── app.py                 # Streamlit UI
├── backend/
│   ├── __init__.py
│   └── backend.py             # LangGraph workflow, SQL + RAG agents, get_bot_response()
├── data/
│   ├── customer_orders.db     # SQLite order data
│   └── Food_Delivery_Policy_final.pdf   # policy doc for RAG
├── .github/workflows/ci.yml   # lint + import check on push
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (local)
```bash
git clone https://github.com/<your-username>/foodhub-chatbot.git
cd foodhub-chatbot
pip install -r requirements.txt
```
Set your Gemini API key as an environment variable:
```bash
export GEMINI_TOKEN=your_key_here   # macOS/Linux
set GEMINI_TOKEN=your_key_here      # Windows
```

## Run
```bash
streamlit run frontend/app.py
```

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io), pointing to `frontend/app.py`.
3. Add `GEMINI_TOKEN` under **App settings → Secrets**.
4. Every push to `main` auto-redeploys the app.

## CI
`.github/workflows/ci.yml` runs on every push to `main`: installs dependencies, lints the code, and smoke-tests that `backend.py` imports cleanly.

## Tech Stack
LangGraph · LangChain · Google Gemini · Chroma (RAG) · SQLite · Streamlit
