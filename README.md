# PlanA — 3-Year ML/AI Learning Platform (Streamlit)

Run a local Streamlit app to navigate a full 3-year curriculum, weekly planners, sprints, and resource checklists.

## Quickstart

```bash
# 1) Create and activate a virtual env (optional but recommended)
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

## Features
- Structured 3-year plan (Year 1–3) with monthly breakdowns
- Interactive 7-day sprint, weekly schedule template, and project tracker
- Progress persistence with Streamlit session state (per session)
- Resource hub with direct links and checklist

## Notes
- This app stores progress in session state only. To persist across sessions, integrate a backend (Supabase/Firebase/SQLite) later.
