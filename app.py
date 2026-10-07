import json
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from services.history_service import load_history, save_session
from services.interview_service import (
    generate_question,
    evaluate_answer,
    summarize_interview,
)


# ---------------------------------------------------------
# GEMINI SETUP
# ---------------------------------------------------------

# Load variables from the .env file (contains GEMINI_API_KEY).
load_dotenv()

MODEL_NAME = "gemini-3.5-flash"

# Maximum length of a typed AI-interview answer. Longer answers are
# rejected locally (never sent to Gemini) to control cost/abuse.
MAX_ANSWER_CHARS = 4000


@st.cache_resource
def get_gemini_client():
    """Create the Gemini client once and reuse it.

    Returns None if the API key is missing, so the app can fall back
    gracefully instead of crashing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_ai_feedback(question, selected_answer, is_correct):
    """Ask Gemini for personalized feedback on a QUIZ answer.

    Returns a tuple: (feedback_text, status). status is one of
    "ok", "quota", "transient", "error". On anything other than "ok"
    the caller keeps showing the standard question-bank explanation, so
    a Gemini failure never breaks the quiz or scoring.
    """
    client = get_gemini_client()
    if client is None:
        return None, "error"

    result_word = "correct" if is_correct else "incorrect"

    prompt = f"""You are a friendly finance interview coach helping a student prepare.

Context:
- Target role: {question['role']}
- Topic: {question['topic']}
- Difficulty: {question['difficulty']}
- Question: {question['question']}
- Correct answer: {question['answer']}
- Student's selected answer: {selected_answer}
- The student's answer was {result_word}.

Give short, encouraging feedback using these four markdown sections exactly:

**Reasoning:** Is the student's reasoning conceptually sound? (1-2 sentences)
**Concept:** A short, clear explanation of the underlying concept. (2-3 sentences)
**Interview tip:** One practical tip for answering this in a real interview. (1 sentence)
**Improvement:** One specific suggestion to improve. (1 sentence)

Keep the whole response under 150 words and beginner-friendly."""

    # Self-contained, status-aware retry (kept local so the quiz feedback
    # path does NOT import newer helper symbols from interview_service; this
    # avoids a boot-time ImportError if that module is cached/stale on a
    # redeploy). Transient 503/network errors are retried with short
    # exponential backoff; a 429 quota error fails fast (retrying it
    # immediately cannot help). Returns (text, status) where status is one
    # of "ok", "quota", "transient", "error". The API key never appears in
    # the prompt or in any logged message.
    last_status = "transient"
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if text:
                return text, "ok"
            last_status = "transient"  # empty response -> retry
        except Exception as error:
            message = str(error).lower()
            code = getattr(error, "code", None)
            if (code == 429 or "resource_exhausted" in message
                    or "quota" in message or "rate limit" in message
                    or "429" in message):
                print(f"[Gemini] quota error: {type(error).__name__}",
                      file=sys.stderr)
                return None, "quota"  # do not retry a quota error
            if (code in (500, 502, 503, 504) or "unavailable" in message
                    or "internal error" in message or "deadline" in message
                    or "timeout" in message or "temporarily" in message
                    or "connection" in message):
                last_status = "transient"
                print(
                    f"[Gemini] transient error attempt {attempt}/3: "
                    f"{type(error).__name__}", file=sys.stderr)
            else:
                print(f"[Gemini] error: {type(error).__name__}",
                      file=sys.stderr)
                return None, "error"  # non-transient; do not spin
        if attempt < 3:
            time.sleep(1.0 * (2 ** (attempt - 1)))

    return None, last_status


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Finance Interview Coach",
    page_icon="💼",
    layout="wide"
)


# ---------------------------------------------------------
# STYLING  (presentation only – a small, self-contained theme)
# ---------------------------------------------------------

def apply_styling():
    """Inject a restrained professional finance/fintech theme.

    Custom coloured blocks (hero, cards) carry their own background and
    text colour so they stay readable in both light and dark Streamlit
    themes. Native headings are left to the theme for safe contrast.
    """
    st.markdown(
        """
        <style>
        :root {
            --navy: #0f2742;
            --accent: #2563eb;
            --muted: #5b6472;
            --line: #e6e9ef;
        }
        /* Consistent, accessible buttons */
        .stButton > button {
            background: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
        }
        .stButton > button:hover { background: #1d4ed8; color: #ffffff; }
        .stButton > button:focus { outline: 2px solid #93c5fd; }

        /* Metric cards with controlled contrast */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16,39,66,.05);
        }
        div[data-testid="stMetric"] label p { color: var(--muted) !important; }
        div[data-testid="stMetricValue"] { color: var(--navy) !important; }

        /* Hero banner (home) */
        .afc-hero {
            background: linear-gradient(135deg, #0f2742 0%, #1e3a5f 100%);
            color: #ffffff;
            padding: 2.4rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
        }
        .afc-hero h1 { color: #ffffff; margin: 0 0 .4rem 0; font-size: 2.1rem; }
        .afc-hero p { color: #d6e2f2; margin: 0; font-size: 1.05rem; }

        /* Feature / info cards */
        .afc-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.3rem 1.2rem;
            box-shadow: 0 1px 3px rgba(16,39,66,.06);
            height: 100%;
        }
        .afc-card .afc-icon { font-size: 1.8rem; }
        .afc-card .afc-title {
            color: var(--navy); font-weight: 700;
            font-size: 1.1rem; margin: .4rem 0 .25rem 0;
        }
        .afc-card .afc-desc { color: var(--muted); font-size: .95rem; }

        /* Sidebar brand */
        .afc-brand {
            color: var(--navy); font-weight: 800; font-size: 1.12rem;
            line-height: 1.25; margin-bottom: .2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon, title, desc):
    st.markdown(
        f'<div class="afc-card"><div class="afc-icon">{icon}</div>'
        f'<div class="afc-title">{title}</div>'
        f'<div class="afc-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# LOAD QUESTION BANK
# ---------------------------------------------------------

@st.cache_data
def load_questions():
    with open("data/questions.json", "r", encoding="utf-8") as file:
        return json.load(file)


questions = load_questions()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

def reset_session():
    """Return QUIZ MODE state to the clean setup screen."""
    st.session_state.quiz_started = False
    st.session_state.session_questions = []
    st.session_state.current_index = 0
    st.session_state.answer_submitted = False
    st.session_state.current_selected = None
    st.session_state.ai_feedback = None
    st.session_state.ai_feedback_status = None
    st.session_state.results = []
    st.session_state.score = 0
    st.session_state.finished = False
    st.session_state.history_saved = False
    st.session_state.history_save_ok = None
    st.session_state.shortfall_message = None


def reset_ai():
    """Return AI INTERVIEW MODE state to the clean setup screen."""
    st.session_state.ai_started = False
    st.session_state.ai_role = None
    st.session_state.ai_difficulty = None
    st.session_state.ai_num_questions = 5
    st.session_state.ai_q_number = 0
    st.session_state.ai_current_question = None
    st.session_state.ai_transcript = []
    st.session_state.ai_qa = []
    st.session_state.ai_total_score = 0
    st.session_state.ai_answer_evaluated = False
    st.session_state.ai_current_feedback = None
    st.session_state.ai_question_error = False
    st.session_state.ai_eval_error = False
    st.session_state.ai_pending_answer = None
    st.session_state.ai_finished = False
    st.session_state.ai_summary = None
    st.session_state.ai_history_saved = False
    st.session_state.ai_history_save_ok = None


if "quiz_started" not in st.session_state:
    reset_session()

if "ai_started" not in st.session_state:
    reset_ai()


def go_to(nav_label):
    """Queue a navigation change to be applied before the sidebar radio
    is built on the next run (avoids modifying a live widget's state)."""
    st.session_state._pending_nav = nav_label
    st.rerun()


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

def render_home():
    st.markdown(
        '<div class="afc-hero"><h1>AI Finance Interview Coach</h1>'
        '<p>Build finance interview confidence through structured practice '
        'and AI feedback.</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_card("📝", "Quiz Practice", "Topic-based finance questions")
    with col2:
        feature_card("🤖", "AI Interview", "Conversational AI interviewer")
    with col3:
        feature_card("📊", "Performance", "Track progress across sessions")

    st.write("")
    left, _ = st.columns([1, 3])
    with left:
        if st.button("Start Practicing", type="primary",
                     use_container_width=True):
            go_to("📝 Quiz Practice")


# ---------------------------------------------------------
# PERFORMANCE DASHBOARD
# ---------------------------------------------------------

def _session_type(session):
    """Older quiz records have no session_type; treat them as Quiz."""
    return session.get("session_type", "Quiz")


def _accuracy_by(sessions, key):
    """Aggregate correct/total across sessions grouped by a top-level
    field (e.g. 'role' or 'difficulty'); returns {group: accuracy%}."""
    agg = {}
    for session in sessions:
        group = session.get(key)
        if group is None:
            continue
        bucket = agg.setdefault(group, {"correct": 0, "total": 0})
        bucket["correct"] += session.get("correct_answers", 0)
        bucket["total"] += session.get("number_of_questions", 0)
    return {
        g: (b["correct"] / b["total"] * 100 if b["total"] else 0)
        for g, b in agg.items()
    }


def _accuracy_by_topic(sessions):
    """Aggregate topic accuracy using each session's topic_performance."""
    agg = {}
    for session in sessions:
        topic_perf = session.get("topic_performance", {}) or {}
        for topic_name, stats in topic_perf.items():
            bucket = agg.setdefault(topic_name, {"correct": 0, "total": 0})
            bucket["correct"] += stats.get("correct", 0)
            bucket["total"] += stats.get("total", 0)
    return {
        t: (b["correct"] / b["total"] * 100 if b["total"] else 0)
        for t, b in agg.items()
    }


def _render_accuracy_block(title, accuracy_map):
    """Show a bar chart plus a readable per-item list for an accuracy map."""
    st.markdown(f"#### {title}")
    if not accuracy_map:
        st.caption("No data yet.")
        return
    frame = pd.DataFrame(
        {"Accuracy %": {k: round(v, 1) for k, v in accuracy_map.items()}}
    ).sort_values("Accuracy %", ascending=False)
    st.bar_chart(frame)
    for name, acc in frame["Accuracy %"].items():
        st.write(f"- **{name}:** {acc:.0f}%")


def render_dashboard():
    st.header("📊 Performance")
    st.caption("Track your progress across quiz and AI interview sessions.")

    history = load_history()

    if not history:
        # Professional empty state with a call to action.
        with st.container(border=True):
            st.subheader("No completed interviews yet.")
            st.write(
                "Complete your first interview to start tracking your "
                "progress."
            )
            if st.button("Start Practicing", type="primary"):
                go_to("📝 Quiz Practice")
        return

    quiz_sessions = [s for s in history if _session_type(s) == "Quiz"]
    ai_sessions = [s for s in history if _session_type(s) == "AI Interview"]

    # ---- Headline metrics ----
    total_questions = sum(s.get("number_of_questions", 0) for s in quiz_sessions)
    total_correct = sum(s.get("correct_answers", 0) for s in quiz_sessions)
    overall_accuracy = (
        total_correct / total_questions * 100 if total_questions else 0
    )
    best_quiz = max(
        (s.get("accuracy_percentage", 0) for s in quiz_sessions), default=0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Interviews", len(history))
    c2.metric("Questions Attempted", total_questions)
    c3.metric("Quiz Sessions", len(quiz_sessions))
    c4.metric("Overall Accuracy", f"{overall_accuracy:.0f}%")
    c5.metric("Best Score", f"{best_quiz:.0f}%")

    st.divider()

    # ---- Quiz analyses ----
    st.subheader("📝 Quiz Analysis")
    if quiz_sessions:
        g1, g2, g3 = st.columns(3)
        with g1:
            with st.container(border=True):
                _render_accuracy_block(
                    "Performance by Topic", _accuracy_by_topic(quiz_sessions)
                )
        with g2:
            with st.container(border=True):
                _render_accuracy_block(
                    "Performance by Role", _accuracy_by(quiz_sessions, "role")
                )
        with g3:
            with st.container(border=True):
                _render_accuracy_block(
                    "Performance by Difficulty",
                    _accuracy_by(quiz_sessions, "difficulty")
                )
    else:
        st.caption("No quiz sessions completed yet.")

    # ---- AI interview analysis ----
    if ai_sessions:
        st.divider()
        st.subheader("🤖 AI Interview Analysis")
        avg_scores = [s.get("average_score", 0) for s in ai_sessions]
        overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
        a1, a2 = st.columns(2)
        a1.metric("AI Interviews Completed", len(ai_sessions))
        a2.metric("Average Score", f"{overall_avg:.1f}/10")

        by_role = {}
        for s in ai_sessions:
            role_name = s.get("role", "Unknown")
            bucket = by_role.setdefault(role_name, [])
            bucket.append(s.get("average_score", 0))
        st.markdown("**Average Score by Role**")
        for role_name, scores in by_role.items():
            avg = sum(scores) / len(scores) if scores else 0
            st.write(f"- **{role_name}:** {avg:.1f}/10")

    st.divider()

    # ---- Recent sessions (all types) ----
    st.subheader("🕒 Recent Sessions")
    recent = list(reversed(history))[:10]
    rows = []
    for s in recent:
        timestamp = str(s.get("timestamp", "")).replace("T", " ")
        stype = _session_type(s)
        if stype == "AI Interview":
            score = f"{s.get('average_score', 0):.1f}/10 avg"
            accuracy = "—"
        else:
            score = f"{s.get('correct_answers', 0)}/" \
                    f"{s.get('number_of_questions', 0)}"
            accuracy = f"{s.get('accuracy_percentage', 0):.0f}%"
        rows.append({
            "Date": timestamp,
            "Type": stype,
            "Role": s.get("role", ""),
            "Topic": s.get("topic", "—"),
            "Difficulty": s.get("difficulty", ""),
            "Score": score,
            "Accuracy": accuracy,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------
# QUIZ MODE
# ---------------------------------------------------------

def render_quiz():

    # ---- SETUP SCREEN ----
    if not st.session_state.quiz_started:

        st.header("📝 Quiz Practice")
        st.caption("Set up a focused practice session.")

        with st.container(border=True):
            st.subheader("Quiz Setup")

            c1, c2 = st.columns(2)
            with c1:
                role = st.selectbox(
                    "Target Role",
                    ["Investment Banking", "Equity Research",
                     "Financial Analyst"]
                )
            with c2:
                topic = st.selectbox(
                    "Topic",
                    [
                        "Accounting", "Valuation", "Financial Markets",
                        "Investment Banking", "Corporate Finance",
                        "Financial Modelling"
                    ]
                )

            c3, c4 = st.columns(2)
            with c3:
                difficulty = st.selectbox(
                    "Difficulty",
                    ["Easy", "Medium", "Hard"]
                )
            with c4:
                num_questions = st.radio(
                    "Number of Questions",
                    [5, 10],
                    index=0,
                    horizontal=True
                )

            start = st.button("Start Quiz", type="primary",
                              use_container_width=True)

        if start:

            filtered_questions = [
                q for q in questions
                if q["role"] == role
                and q["topic"] == topic
                and q["difficulty"] == difficulty
            ]

            if not filtered_questions:
                st.warning(
                    "No questions are currently available for this "
                    "combination. Try another topic or difficulty."
                )
            else:
                requested = num_questions
                available = len(filtered_questions)
                session_size = min(requested, available)

                chosen = random.sample(filtered_questions, session_size)

                # Shuffle each question's options ONCE here, at session
                # initialization, storing COPIES so the cached bank is never
                # mutated and the order stays stable across reruns.
                selected = []
                for q in chosen:
                    q_copy = dict(q)
                    q_copy["options"] = random.sample(
                        q["options"], len(q["options"])
                    )
                    selected.append(q_copy)

                reset_session()
                st.session_state.quiz_started = True
                st.session_state.session_questions = selected

                if session_size < requested:
                    st.session_state.shortfall_message = (
                        f"Only {available} question(s) are available for this "
                        f"combination, so this session has {session_size} "
                        f"instead of {requested}."
                    )
                else:
                    st.session_state.shortfall_message = None

                st.rerun()

        return

    # ---- RESULTS SCREEN ----
    if st.session_state.finished:

        st.header("🏁 Quiz Results")

        results = st.session_state.results
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        incorrect = total - correct
        accuracy = (correct / total * 100) if total else 0

        topic_stats = {}
        for r in results:
            stats = topic_stats.setdefault(
                r["topic"], {"correct": 0, "total": 0}
            )
            stats["total"] += 1
            if r["is_correct"]:
                stats["correct"] += 1

        # Save this quiz session to history EXACTLY ONCE.
        if not st.session_state.history_saved:
            meta = (
                st.session_state.session_questions[0]
                if st.session_state.session_questions
                else {}
            )
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "session_type": "Quiz",
                "role": meta.get("role"),
                "topic": meta.get("topic"),
                "difficulty": meta.get("difficulty"),
                "number_of_questions": total,
                "correct_answers": correct,
                "incorrect_answers": incorrect,
                "score": correct,
                "accuracy_percentage": round(accuracy, 1),
                "topic_performance": topic_stats,
                "question_results": [
                    {
                        "question": r["question"],
                        "selected": r["selected"],
                        "correct_answer": r["correct_answer"],
                        "is_correct": r["is_correct"],
                        "topic": r["topic"],
                    }
                    for r in results
                ],
            }
            st.session_state.history_save_ok = save_session(record)
            st.session_state.history_saved = True

        if st.session_state.history_save_ok is False:
            st.warning(
                "⚠️ Your results are shown below, but this session could not "
                "be saved to history."
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Score", f"{correct} / {total}")
        col2.metric("Accuracy", f"{accuracy:.0f}%")
        col3.metric("Questions Completed", total)

        st.progress(int(accuracy))

        st.divider()

        with st.container(border=True):
            st.subheader("📊 Performance by Topic")
            for topic_name, stats in topic_stats.items():
                pct = stats["correct"] / stats["total"] * 100
                st.write(
                    f"- **{topic_name}:** {stats['correct']} / "
                    f"{stats['total']} ({pct:.0f}%)"
                )

        with st.expander("Review each question"):
            for i, r in enumerate(results, start=1):
                mark = "✅" if r["is_correct"] else "❌"
                st.markdown(f"**{mark} Q{i}. {r['question']}**")
                st.write(f"Your answer: {r['selected']}")
                if not r["is_correct"]:
                    st.write(f"Correct answer: {r['correct_answer']}")
                st.divider()

        if st.button("🔄 Start New Quiz", type="primary"):
            reset_session()
            st.rerun()

        return

    # ---- QUIZ SCREEN (one question at a time) ----
    session_questions = st.session_state.session_questions
    total = len(session_questions)
    index = st.session_state.current_index
    question = session_questions[index]

    if st.session_state.get("shortfall_message"):
        st.info(st.session_state.shortfall_message)

    st.header(f"Question {index + 1} of {total}")
    st.progress((index + 1) / total)

    with st.container(border=True):
        st.caption(
            f"{question['role']}  •  "
            f"{question['topic']}  •  "
            f"{question['difficulty']}"
        )
        st.markdown(f"### {question['question']}")

        if not st.session_state.answer_submitted:

            selected_answer = st.radio(
                "Select your answer:",
                question["options"],
                key=f"answer_{index}_{question['id']}"
            )

            submit = st.button("✅ Submit Answer", type="primary")

            if submit:

                st.session_state.answer_submitted = True
                st.session_state.current_selected = selected_answer

                is_correct = selected_answer == question["answer"]

                if is_correct:
                    st.session_state.score += 1

                st.session_state.results.append({
                    "id": question["id"],
                    "topic": question["topic"],
                    "question": question["question"],
                    "selected": selected_answer,
                    "correct_answer": question["answer"],
                    "is_correct": is_correct,
                })

                with st.spinner("Getting AI coach feedback..."):
                    feedback, status = get_ai_feedback(
                        question, selected_answer, is_correct
                    )

                st.session_state.ai_feedback = feedback
                st.session_state.ai_feedback_status = status

                st.rerun()

            return

        # Answer already submitted -> show locked choice + result.
        selected_answer = st.session_state.current_selected

        st.radio(
            "Your answer (locked):",
            question["options"],
            index=question["options"].index(selected_answer),
            key=f"answer_locked_{index}_{question['id']}",
            disabled=True
        )

        if selected_answer == question["answer"]:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect")
            st.write(f"**Correct answer:** {question['answer']}")

        st.info(f"**Explanation:** {question['explanation']}")

    # Feedback shown in its own card below the question card. The standard
    # explanation above is ALWAYS shown, so a Gemini failure never removes
    # useful content — it only affects this optional AI coaching card.
    if st.session_state.answer_submitted:
        status = st.session_state.ai_feedback_status
        with st.container(border=True):
            if status == "ok" and st.session_state.ai_feedback:
                st.markdown("### 🤖 AI Coach Feedback")
                st.markdown(st.session_state.ai_feedback)
            elif status == "quota":
                # Quota/rate limit: retrying now cannot help, so no retry
                # button is offered (avoids obviously-failing retries).
                st.warning(
                    "⚠️ AI Coach quota is temporarily unavailable. Your quiz "
                    "result and standard explanation are still available."
                )
            else:
                # Transient / network / other: a retry may well succeed.
                st.warning(
                    "⚠️ AI Coach is temporarily unavailable. Your answer has "
                    "been recorded and the standard explanation is still "
                    "available. You can retry AI feedback in a moment."
                )
                if st.button("🔄 Retry AI Feedback"):
                    sel = st.session_state.current_selected
                    is_corr = sel == question["answer"]
                    with st.spinner("Retrying AI feedback..."):
                        fb, stt = get_ai_feedback(question, sel, is_corr)
                    st.session_state.ai_feedback = fb
                    st.session_state.ai_feedback_status = stt
                    st.rerun()

        st.metric("Current Score", st.session_state.score)

        is_last = index + 1 >= total
        button_label = "🏁 See Results" if is_last else "➡️ Next Question"

        if st.button(button_label, type="primary"):
            if is_last:
                st.session_state.finished = True
            else:
                st.session_state.current_index += 1
                st.session_state.answer_submitted = False
                st.session_state.current_selected = None
                st.session_state.ai_feedback = None
                st.session_state.ai_feedback_status = None
            st.rerun()


# ---------------------------------------------------------
# AI INTERVIEW MODE
# ---------------------------------------------------------

def render_ai_results(client):
    st.header("🎯 AI Interview Results")

    qa = st.session_state.ai_qa
    completed = len(qa)
    total_score = st.session_state.ai_total_score
    average = (total_score / completed) if completed else 0

    # Save this AI interview to history EXACTLY ONCE.
    if not st.session_state.ai_history_saved:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_type": "AI Interview",
            "role": st.session_state.ai_role,
            "difficulty": st.session_state.ai_difficulty,
            "number_of_questions": completed,
            "average_score": round(average, 2),
            "final_overall_score": total_score,
            "per_question_scores": [q.get("score", 0) for q in qa],
        }
        st.session_state.ai_history_save_ok = save_session(record)
        st.session_state.ai_history_saved = True

    if st.session_state.ai_history_save_ok is False:
        st.warning(
            "⚠️ Your results are shown below, but this interview could not "
            "be saved to history."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Score", f"{total_score}")
    c2.metric("Average Score", f"{average:.1f} / 10")
    c3.metric("Questions Completed", completed)

    summary = st.session_state.ai_summary

    if summary:
        if summary.get("overall_feedback"):
            with st.container(border=True):
                st.subheader("🧭 Overall Feedback")
                st.write(summary.get("overall_feedback", ""))

        col_s, col_i = st.columns(2)
        with col_s:
            with st.container(border=True):
                st.subheader("💪 Strengths")
                if summary.get("strengths"):
                    for item in summary["strengths"]:
                        st.write(f"- {item}")
                else:
                    st.caption("—")
        with col_i:
            with st.container(border=True):
                st.subheader("🔧 Areas to Improve")
                if summary.get("improvements"):
                    for item in summary["improvements"]:
                        st.write(f"- {item}")
                else:
                    st.caption("—")

        if summary.get("recommendations"):
            with st.container(border=True):
                st.subheader("✅ Recommendations")
                for i, rec in enumerate(summary["recommendations"], start=1):
                    st.write(f"{i}. {rec}")
    else:
        st.info(
            "The AI overall summary is temporarily unavailable, but your "
            "per-question scores are shown below."
        )
        if st.button("🔄 Retry Summary"):
            with st.spinner("Summarizing your interview..."):
                summ, ok = summarize_interview(
                    client, MODEL_NAME,
                    st.session_state.ai_role,
                    st.session_state.ai_difficulty,
                    qa,
                )
            if ok:
                st.session_state.ai_summary = summ
            st.rerun()

    st.divider()

    with st.container(border=True):
        st.subheader("📋 Question-by-Question Scores")
        rows = []
        for i, q in enumerate(qa, start=1):
            rows.append({
                "Q": i,
                "Score": f"{q.get('score', 0)}/10",
                "Question": q.get("question", ""),
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True
            )

    if st.button("🔄 Start New AI Interview", type="primary"):
        reset_ai()
        st.rerun()


def render_ai_interview():
    st.header("🤖 AI Interview")

    client = get_gemini_client()
    if client is None:
        st.error(
            "AI Interview Mode needs a configured Gemini API key. Please set "
            "GEMINI_API_KEY in your .env file."
        )
        return

    role = st.session_state.ai_role
    difficulty = st.session_state.ai_difficulty
    num_questions = st.session_state.ai_num_questions

    # ---- SETUP SCREEN ----
    if not st.session_state.ai_started:
        st.caption(
            "A Gemini-powered interviewer asks finance questions one at a "
            "time, evaluates your typed answers, and adapts follow-ups to "
            "what you say."
        )

        with st.container(border=True):
            st.subheader("Interview Setup")

            c1, c2 = st.columns(2)
            with c1:
                chosen_role = st.selectbox(
                    "Target Role",
                    ["Investment Banking", "Equity Research",
                     "Financial Analyst"],
                    key="ai_role_select"
                )
            with c2:
                chosen_difficulty = st.selectbox(
                    "Difficulty",
                    ["Easy", "Medium", "Hard"],
                    key="ai_difficulty_select"
                )

            chosen_num = st.radio(
                "Number of Questions",
                [5, 10],
                index=0,
                horizontal=True,
                key="ai_num_select"
            )

            start = st.button("Start AI Interview", type="primary",
                              use_container_width=True)

        if start:
            reset_ai()
            st.session_state.ai_started = True
            st.session_state.ai_role = chosen_role
            st.session_state.ai_difficulty = chosen_difficulty
            st.session_state.ai_num_questions = chosen_num
            st.session_state.ai_q_number = 1

            # Generate the first question here (in the click handler) so a
            # rerun never generates a second question by accident.
            with st.spinner("The interviewer is preparing your first "
                            "question..."):
                question, ok = generate_question(
                    client, MODEL_NAME, chosen_role, chosen_difficulty,
                    chosen_num, 1, []
                )
            if ok:
                st.session_state.ai_current_question = question
                st.session_state.ai_transcript.append(
                    {"role": "interviewer", "content": question}
                )
            else:
                st.session_state.ai_question_error = True
            st.rerun()

        return

    # ---- RESULTS SCREEN ----
    if st.session_state.ai_finished:
        render_ai_results(client)
        return

    k = st.session_state.ai_q_number

    # Progress reflects how many answers are completed.
    completed = len(st.session_state.ai_qa)
    st.caption(f"**Role:** {role}   |   **Difficulty:** {difficulty}")
    st.subheader(f"Question {k} of {num_questions}")
    st.progress(completed / num_questions)

    # ---- QUESTION GENERATION FAILED: offer a retry ----
    if st.session_state.ai_current_question is None:
        st.error(
            "⚠️ The interviewer couldn't generate a question just now "
            "(the AI service may be temporarily busy)."
        )
        if st.button("🔄 Retry Question", type="primary"):
            with st.spinner("Retrying..."):
                question, ok = generate_question(
                    client, MODEL_NAME, role, difficulty, num_questions,
                    k, st.session_state.ai_transcript
                )
            if ok:
                st.session_state.ai_current_question = question
                st.session_state.ai_question_error = False
                st.session_state.ai_transcript.append(
                    {"role": "interviewer", "content": question}
                )
            st.rerun()
        return

    # ---- Conversation so far, rendered as chat bubbles ----
    for turn in st.session_state.ai_transcript:
        if turn["role"] == "interviewer":
            with st.chat_message("assistant", avatar="🧑‍💼"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(turn["content"])

    # ---- ANSWERING (not yet evaluated) ----
    if not st.session_state.ai_answer_evaluated:

        with st.container(border=True):
            st.markdown("**Your Answer**")
            answer = st.text_area(
                "Your answer:",
                key=f"ai_answer_{k}",
                height=160,
                label_visibility="collapsed",
                placeholder="Type your answer as you would explain it in an "
                            "interview..."
            )
            st.caption(f"Up to {MAX_ANSWER_CHARS:,} characters.")

            if st.session_state.ai_eval_error:
                st.warning(
                    "⚠️ The evaluation didn't come through. Your answer is "
                    "kept — click Submit Answer to try again."
                )

            submit = st.button("✅ Submit Answer", type="primary",
                               use_container_width=True)

        if submit:
            typed = (answer or "").strip()
            if not typed:
                # Empty / whitespace-only: do NOT call Gemini or advance.
                st.warning("Please enter an answer before submitting.")
            elif len(typed) > MAX_ANSWER_CHARS:
                # Too long: reject locally, keep state, do NOT call Gemini.
                st.warning(
                    f"Your answer is too long ({len(typed):,} characters). "
                    f"Please shorten it to under {MAX_ANSWER_CHARS:,} "
                    "characters and submit again."
                )
            else:
                st.session_state.ai_pending_answer = typed
                with st.spinner("The interviewer is evaluating your "
                                "answer..."):
                    evaluation, ok = evaluate_answer(
                        client, MODEL_NAME, role, difficulty,
                        st.session_state.ai_current_question, typed
                    )
                if ok:
                    st.session_state.ai_eval_error = False
                    st.session_state.ai_current_feedback = evaluation
                    st.session_state.ai_total_score += evaluation["score"]
                    st.session_state.ai_qa.append({
                        "question": st.session_state.ai_current_question,
                        "answer": typed,
                        **evaluation,
                    })
                    # Add the answer to the transcript so the next question
                    # can reference it (genuine multi-turn memory).
                    st.session_state.ai_transcript.append(
                        {"role": "candidate", "content": typed}
                    )
                    st.session_state.ai_answer_evaluated = True
                else:
                    st.session_state.ai_eval_error = True
                st.rerun()

        return

    # ---- FEEDBACK (answer evaluated) ----
    evaluation = st.session_state.ai_current_feedback

    with st.container(border=True):
        st.markdown("### 📝 Evaluation")
        st.metric("Score", f"{evaluation['score']} / 10")
        st.markdown(f"**✅ What you did well:** {evaluation['did_well']}")
        st.markdown(
            f"**🔍 What to improve:** {evaluation['missing']}"
        )
        st.markdown(f"**📘 Correct concept:** {evaluation['correct_concept']}")
        st.markdown(f"**💡 Interview tip:** {evaluation['tip']}")

    st.metric("Total Score So Far", f"{st.session_state.ai_total_score}")

    is_last = k >= num_questions
    label = "🎯 See Interview Results" if is_last else "➡️ Next Question"

    if st.button(label, type="primary"):
        if is_last:
            with st.spinner("Preparing your final interview report..."):
                summary, ok = summarize_interview(
                    client, MODEL_NAME, role, difficulty,
                    st.session_state.ai_qa
                )
            st.session_state.ai_summary = summary if ok else None
            st.session_state.ai_finished = True
        else:
            st.session_state.ai_q_number += 1
            st.session_state.ai_answer_evaluated = False
            st.session_state.ai_current_feedback = None
            st.session_state.ai_pending_answer = None
            # Generate the next question using the full conversation so far.
            with st.spinner("The interviewer is preparing the next "
                            "question..."):
                question, ok = generate_question(
                    client, MODEL_NAME, role, difficulty, num_questions,
                    st.session_state.ai_q_number, st.session_state.ai_transcript
                )
            if ok:
                st.session_state.ai_current_question = question
                st.session_state.ai_transcript.append(
                    {"role": "interviewer", "content": question}
                )
            else:
                st.session_state.ai_current_question = None
                st.session_state.ai_question_error = True
        st.rerun()


# ---------------------------------------------------------
# NAVIGATION + ROUTING
# ---------------------------------------------------------

apply_styling()

# Apply any queued navigation BEFORE the sidebar radio is instantiated.
if "nav" not in st.session_state:
    st.session_state.nav = "🏠 Home"
if st.session_state.get("_pending_nav") is not None:
    st.session_state.nav = st.session_state._pending_nav
    st.session_state._pending_nav = None

st.sidebar.markdown(
    '<div class="afc-brand">💼 AI Finance Interview Coach</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Practice finance interviews with AI-powered feedback.")
st.sidebar.divider()

nav = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📝 Quiz Practice", "🤖 AI Interview", "📊 Performance"],
    key="nav",
    label_visibility="collapsed",
)

if nav == "🏠 Home":
    render_home()
elif nav == "📝 Quiz Practice":
    render_quiz()
elif nav == "🤖 AI Interview":
    render_ai_interview()
else:
    render_dashboard()
