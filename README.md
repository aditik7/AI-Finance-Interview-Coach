# AI Finance Interview Coach

An AI-powered finance interview preparation application built with Streamlit
and the Google Gemini API. It combines a structured finance quiz with a
conversational AI interviewer that asks questions, evaluates answers, and
tracks progress across sessions.

---

## 1. Problem Statement

Candidates preparing for finance interviews (Investment Banking, Equity
Research, Financial Analyst) often struggle to practice realistically. Static
question banks give no feedback, generic chatbots are not tailored to finance
interviews, and mock interviews with mentors are hard to schedule and not
repeatable. Learners rarely get *immediate, structured, interview-style
feedback* on both *what* they answered and *how* they explained it.

## 2. Objective

The system is designed to help users build finance interview confidence
through repeatable, structured practice with instant AI feedback. It lets a
candidate drill topic-specific questions, experience a realistic multi-turn AI
interview, and review measurable performance over time — all in one place.

## 3. Target Users

- Finance students
- MBA / PGDM students
- Candidates preparing for **Investment Banking**, **Equity Research**, and
  **Financial Analyst** interviews

## 4. Key Features

- **Role-based finance quiz** (Investment Banking, Equity Research, Financial Analyst)
- **Topic and difficulty selection** (6 topics; Easy / Medium / Hard)
- **Randomized questions and answer options** (shuffled once per session)
- **Instant scoring** (correct / incorrect)
- **Standard explanations** from the question bank
- **Gemini AI feedback** on each quiz answer
- **AI Interview Mode** — a conversational interviewer powered by Gemini
- **Multi-turn interview evaluation** with per-answer scoring (0–10)
- **Score tracking** and a final interview report (strengths, improvements, recommendations)
- **Persistent session history** stored in a local JSON file
- **Performance dashboard** (accuracy by topic / role / difficulty, recent sessions)
- **Input validation and AI guardrails** (see section 12)

## 5. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.13 |
| UI | Streamlit |
| AI model | Google Gemini (`gemini-3.5-flash`) |
| Gemini SDK | `google-genai` |
| Data handling | Pandas |
| Config / secrets | `python-dotenv` (`.env`) |
| Storage | JSON files (question bank + session history) |

## 6. System Architecture

```
          ┌──────────────────────────────────────────────────────────┐
          │                        User                               │
          └───────────────────────────┬──────────────────────────────┘
                                       │ interacts with
                          ┌────────────▼────────────┐
                          │      Streamlit UI        │  (app.py)
                          │  Home / Quiz / Interview │
                          │  / Performance           │
                          └────────────┬────────────┘
                                       │ calls
                          ┌────────────▼────────────┐
                          │    Application Logic     │
                          │  quiz flow, AI interview │
                          │  state, scoring          │
                          └───┬───────────────┬──────┘
                reads/writes  │               │  calls (services/)
             ┌────────────────▼───┐      ┌────▼─────────────────────┐
             │  Question Bank      │      │   Gemini API             │
             │  data/questions.json│      │  (google-genai client)   │
             │  Session History    │      │  question gen / evaluate │
             │  data/session_      │      │  / summarize / feedback  │
             │  history.json       │      └────┬─────────────────────┘
             └─────────────────────┘           │ returns feedback
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                          ┌──────────────────────┐
                          │  Performance Dashboard │
                          │  (aggregated history)  │
                          └──────────────────────┘
```

**Flow:** User → Streamlit UI → Application Logic → Question Bank / Session
History → Gemini API → Feedback → Dashboard.

## 7. Project Structure

```
AI-Finance-Interview-Coach/
├── app.py                      # Main Streamlit app (UI, routing, quiz & AI interview flow)
├── gemini_test.py              # Standalone script to verify the Gemini API connection
├── requirements.txt            # Pinned direct dependencies
├── .env                        # Local secrets (GEMINI_API_KEY) — NOT committed
├── .gitignore                  # Excludes .env, caches, venv, local history
├── README.md                   # This file
├── data/
│   └── questions.json          # Finance question bank (56 questions)
│                               # session_history.json is created here at runtime
└── services/
    ├── __init__.py
    ├── history_service.py      # Safe load/save of session history (JSON)
    └── interview_service.py    # Gemini logic: question generation, evaluation, summary
```

> `data/session_history.json` and `__pycache__/` are generated automatically
> and are intentionally git-ignored.

## 8. Setup Instructions

**Prerequisites:** Python 3.13 (the project was developed and tested on 3.13).

```bash
# 1. Clone / download the project, then open a terminal in the project folder.

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1
#    Windows (cmd):
.venv\Scripts\activate.bat
#    macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create the .env file (see section 9) and add your Gemini API key.

# 6. Run the application
streamlit run app.py
```

The app opens in your browser (default `http://localhost:8501`).

## 9. Gemini API Configuration

The application reads the Gemini API key from an environment variable named
`GEMINI_API_KEY`, loaded from a local `.env` file via `python-dotenv`.

Create a file named `.env` in the project root with this single line:

```
GEMINI_API_KEY=your_api_key_here
```

- Obtain a key from Google AI Studio and paste it after `GEMINI_API_KEY=`.
- The key is read only inside `get_gemini_client()` and passed directly to the
  Gemini client. It is **never** displayed in the UI, included in prompts,
  written to session history, or printed in logs.
- **`.env` must never be committed to version control.** It is already listed
  in `.gitignore`. If the key is missing, the app degrades gracefully instead
  of crashing.

## 10. Application Workflow

### Quiz Practice
1. Select target role, topic, difficulty, and number of questions (5 or 10).
2. The app filters the question bank and selects unique questions; each
   question's options are shuffled once for the session.
3. For each question: choose an answer and submit. The answer is locked (it
   cannot be changed or resubmitted).
4. The app shows correct/incorrect, the standard explanation, and Gemini AI
   feedback (with graceful fallback if the AI is unavailable).
5. After the final question, a results screen shows score, accuracy, and
   topic performance; the session is saved to history exactly once.

### AI Interview
1. Select role, difficulty, and number of questions.
2. A Gemini-powered interviewer asks one finance question at a time in a
   chat-style interface.
3. You type an answer and submit; Gemini evaluates it and returns a structured
   card: **Score (0–10)**, what you did well, what to improve, the correct
   concept, and an interview tip.
4. The interviewer uses the running conversation as context, so follow-up
   questions can build on your previous answers (multi-turn memory).
5. After the final question, a results screen shows overall and average score,
   strengths, areas to improve, and recommendations; the interview is saved to
   history exactly once.

## 11. AI Guardrails

The application implements the following protections (see `app.py` and
`services/interview_service.py`):

- **Empty input validation** — empty/whitespace answers are rejected locally;
  Gemini is not called and the interview does not advance.
- **4000-character limit** — overly long answers are rejected locally (never
  sent to Gemini), preserving the current interview state.
- **Prompt-injection resistance** — the interviewer and evaluator prompts
  instruct the model to treat candidate text purely as an answer, to refuse
  attempts to change its role or reveal system instructions / API keys, and
  the untrusted answer is clearly delimited for the evaluator.
- **Off-topic handling** — unrelated inputs are politely redirected back to the
  finance interview, and off-topic answers are scored low.
- **Duplicate submission prevention** — session-state flags
  (`answer_submitted`, `ai_answer_evaluated`, `history_saved`,
  `ai_history_saved`) ensure an answer is counted once, a question is generated
  once, and a session is saved to history exactly once despite Streamlit
  reruns.
- **API failure handling** — Gemini calls use a retry loop and return a
  success flag; on failure the user's answer is preserved and a retry is
  offered, and quiz feedback falls back to the standard explanation. The app
  never crashes due to an AI/API error.
- **API key protection** — the key is confined to `get_gemini_client()` and
  never appears in the UI, prompts, history, or logs.
- **Graceful handling of malformed/empty history** — a missing, empty, or
  malformed `session_history.json` is treated as "no history" and never
  crashes the dashboard.

## 12. Testing

The following tests were performed during development:

- **`py_compile`** on `app.py` and the `services/` modules — no syntax errors.
- **Headless Streamlit boot** (`streamlit run app.py --server.headless true`)
  — the app serves successfully (HTTP 200) with no startup errors.
- **AppTest page rendering** (`streamlit.testing.v1.AppTest`) — Home, Quiz
  Practice, AI Interview, and Performance pages all render with **zero
  exceptions**.
- **Quiz flow** — starting a quiz selects the expected number of unique
  questions, and submitting an answer records the result and advances
  correctly.
- **Gemini feedback** — verified to return real feedback live for quiz answers
  and AI interview evaluation (subject to the free-tier quota; see
  Limitations).
- **Randomized options** — statistically verified that the correct answer
  appears in all four positions and that the original question bank is never
  mutated.

## 13. Limitations

- **Gemini free-tier quota** — the free tier allows a limited number of
  requests per day (approximately 20/day for `gemini-3.5-flash`). Because an
  AI interview makes several calls per session, only a small number of AI
  interviews can be run per day on the free tier. Exceeding the quota returns
  an HTTP 429 error, which the app handles gracefully (retry / fallback), but
  AI feedback may be temporarily unavailable until the quota resets.
- **JSON-based persistence** — session history is stored in a single local
  JSON file. This is simple and portable but is single-user, not concurrent,
  and not suitable for large-scale or multi-user deployment.
- **Question bank size** — the bundled bank contains 56 questions, so some
  role/topic/difficulty combinations have fewer than the requested number of
  questions (handled gracefully by reducing the session size).

## 14. Future Scope

- Larger and continuously expanded question bank
- Database storage (e.g., SQLite / PostgreSQL) in place of JSON
- User authentication and per-user progress
- Personalized learning paths based on weak areas
- Richer analytics and trend visualizations
- Additional finance roles and specializations
- Voice-based interview mode (speech input/output)

## 15. Academic Project Note

This is an AI-powered finance interview preparation application developed as an
academic project. It demonstrates practical integration of a large language
model (Google Gemini) into an interactive Streamlit application, including
structured prompting, multi-turn conversation, input validation, AI
guardrails, graceful error handling, and local data persistence.
