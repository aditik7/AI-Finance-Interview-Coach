"""Simple, file-based persistence for completed interview sessions.

All functions are defensive: a missing, empty, or malformed history file
never raises — it is treated as "no history yet" — and a failed save
returns False instead of crashing the app. No API keys or secrets are
ever stored here; only quiz results are saved.
"""

import json
import os

# History lives next to the question bank, in the data/ folder.
HISTORY_PATH = os.path.join("data", "session_history.json")


def load_history():
    """Return the list of saved session records.

    Handles every failure gracefully:
    - file does not exist  -> []
    - file is empty        -> []
    - file is malformed    -> []
    - file is not a list   -> []
    """
    if not os.path.exists(HISTORY_PATH):
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            return []

        data = json.loads(content)
        if not isinstance(data, list):
            return []

        return data

    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupt or unreadable history must not crash the app.
        return []


def save_session(record):
    """Append one session record to the history file.

    Creates the data/ folder and the file automatically if needed.
    Returns True on success, False on any failure (never raises).
    """
    try:
        folder = os.path.dirname(HISTORY_PATH)
        if folder:
            os.makedirs(folder, exist_ok=True)

        history = load_history()
        history.append(record)

        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)

        return True

    except (OSError, TypeError, ValueError):
        return False
