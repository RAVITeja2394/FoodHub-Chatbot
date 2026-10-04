# FoodHub ChatBot 🍔

A multi-turn customer support chatbot for a food delivery platform, built with **Streamlit** (frontend) and **LangGraph** (backend). It uses **Groq** and **Google Gemini** models, with Groq tried first when a key is configured and Gemini as the fallback. It answers order status questions from a SQLite database, handles refund and payment issues, and answers policy questions with RAG over the delivery policy PDF.

## Features

- Customer selection from a dropdown (C1011–C1015) or by typing a Customer ID
- Multi-turn conversations with in-memory session memory (LangGraph `MemorySaver`; history is lost when the app restarts)
- Intent classification that routes each message to the right handler:
  - **STATUS** — order status, delivery tracking, and order contents (SQL lookup)
  - **POLICY** — cancellation and policy questions (RAG over the policy PDF)
  - **REFUND_ELIGIBILITY** — checks lateness against the 30-minute threshold (25% refund when more than 30 minutes late)
  - **REFUND_STATUS_CHECK** — refund status for cancelled orders
  - **PAYMENT_FAILURE** — standard payment-failure response
  - **ESCALATION** — raises a service ticket (SRxxxx) for human follow-up
  - **MALICIOUS / OUT_OF_SCOPE** — blocked with a generic refusal
  - **NOT_CLEAR** — asks a clarifying question
- Customer-scoped, read-only SQL: only a single `SELECT` on the `orders` table is allowed, it must filter on the session's `cust_id`, and the database is opened read-only
- Relevance and groundedness checks on policy answers, with a service ticket raised when the answer can't be trusted
- Quick-question shortcuts, 👍/👎 buttons on replies, a clear-chat button, and a downloadable chat transcript
- Session ends when the user types an exit keyword (`exit`, `quit`, `thank you`, `fine`, `got it, thank you`) or clicks **End**

## Workflow

```
START → classifier_agent → router
  ├─ STATUS / REFUND_*   → sql_agent → post_sql_router
  │                           ├─ REFUND_STATUS_CHECK → refund_status_handler ─┐
  │                           ├─ REFUND_ELIGIBILITY  → refund_eligibility_handler ─┤
  │                           └─ STATUS              ─────────────────────────────┤
  ├─ POLICY              → rag_agent ────────────────────────────────────────────┤
  │                                                                              ↓
  │                                                                       formatter_node → END
  ├─ PAYMENT_FAILURE     → payment_failure_node → END
  ├─ ESCALATION          → escalation_agent → END
  ├─ MALICIOUS / OUT_OF_SCOPE → blocked_response_node → END
  └─ NOT_CLEAR           → clarify_node → END
```

The SQL agent, the refund handlers, and the RAG node each put their result in the graph state, and `formatter_node` writes the final customer-facing reply. For policy answers, the formatter also runs a groundedness check.

## Project Structure

```
foodhub-chatbot/
├── frontend/
│   └── app.py                 # Streamlit UI
├── backend/
│   └── backend.py             # LangGraph workflow, SQL + RAG nodes, get_bot_response()
├── data/
│   ├── customer_orders.db               # SQLite order data
│   └── Food_Delivery_Policy_final.pdf   # policy document used for RAG
├── requirements.txt
└── README.md
```

The vector index is built from the policy PDF at startup and stored in `data/chroma_policy_db/`. It is generated automatically, so it doesn't need to be committed.

## Setup (local)

```bash
git clone https://github.com/<your-username>/foodhub-chatbot.git
cd foodhub-chatbot
pip install -r requirements.txt
```

### API keys

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_TOKEN` | Yes | Gemini API key. The backend raises an error on import if it is missing. |
| `GROQ_API_KEY` | No | Groq API key. When set, Groq models are tried first and Gemini is the fallback. |
| `GROQ_API_KEY2` | No | Second Groq account key, tried after the first account. |

macOS / Linux:
```bash
export GEMINI_TOKEN=your_key_here
export GROQ_API_KEY=your_key_here   # optional
```

Windows (cmd):
```
set GEMINI_TOKEN=your_key_here
set GROQ_API_KEY=your_key_here
```

Optional environment variables for changing the model chains and rate limits (`GEMINI_MODEL_CHAIN`, `GROQ_MODEL_CHAIN`, `GROQ_TPM_LIMIT`, `GEMINI_RPM_LIMIT`, and others) are defined near the top of `backend/backend.py`, along with their defaults.

## Run

```bash
streamlit run frontend/app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io) pointing to `frontend/app.py`.
3. Under **App settings → Secrets**, add:
   ```toml
   GEMINI_TOKEN = "your_key_here"
   GROQ_API_KEY = "your_key_here"      # optional
   GROQ_API_KEY2 = "your_key_here"     # optional
   ```
4. Every push to `main` redeploys the app.

## Tech Stack

LangGraph · LangChain · Groq · Google Gemini · Chroma (RAG) · SQLite · Streamlit
