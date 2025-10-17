import datetime as dt
from dateutil.relativedelta import relativedelta
from typing import Dict, List

import pandas as pd
import streamlit as st


# ----------------------
# Session State Helpers
# ----------------------
PROGRESS_KEYS = {
    "y1": "progress_year1",
    "y2": "progress_year2",
    "y3": "progress_year3",
    "portfolio": "progress_portfolio",
    "resources": "progress_resources",
    "sprint": "sprint_tasks",
    "weekly": "weekly_plan",
}


def ensure_session_state_defaults() -> None:
    if PROGRESS_KEYS["y1"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["y1"]] = {}
    if PROGRESS_KEYS["y2"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["y2"]] = {}
    if PROGRESS_KEYS["y3"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["y3"]] = {}
    if PROGRESS_KEYS["portfolio"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["portfolio"]] = {}
    if PROGRESS_KEYS["resources"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["resources"]] = {}
    if PROGRESS_KEYS["sprint"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["sprint"]] = default_sprint_tasks()
    if PROGRESS_KEYS["weekly"] not in st.session_state:
        st.session_state[PROGRESS_KEYS["weekly"]] = default_weekly_template()


# ----------------------
# Data Definitions
# ----------------------
YEAR1_MONTHLY = [
    {
        "title": "Oy 1–2: Matematika & Python basics",
        "items": [
            "Math: Precalculus / Calculus I (Khan Academy yoki MIT OCW)",
            "Python: sintaksis, funksiyalar, list/dict, virtualenv",
            "Amaliy: 10 mini kod mashqlari + GitHub repo",
        ],
        "weeks": [
            "Hafta 1: Precalculus reviziya; Python sintaksis, types, input/output",
            "Hafta 2: Calculus I — limits/derivatives; Python functions & modules",
            "Hafta 3: Derivatives qo‘llanmalar; data structures (list/dict/set)",
            "Hafta 4: Integrals kirish; venv, pip, basic file I/O + 3 mini-mashq",
            "Hafta 5: Applications of integrals; exceptions, logging + 3 mini-mashq",
            "Hafta 6: Series/approx; CLI tools, argparse, notebooks + 4 mini-mashq",
            "Hafta 7: Git/GitHub: repo, branching, PR flow; first README",
            "Hafta 8: Repeat & consolidation; small EDA on CSV (pandas kirish)",
        ],
    },
    {
        "title": "Oy 3–4: Linear Algebra + CS50 start",
        "items": [
            "Linear Algebra (MIT OCW, Gilbert Strang)",
            "CS50 boshlash — algoritmlar, C va Python",
            "Amaliy: vektor/matrix kodlash, notebooklar",
        ],
        "weeks": [
            "Hafta 1: Vectors, linear combinations; NumPy basics (faqat tekshirish)",
            "Hafta 2: Matrices, matrix multiplication; CS50 Lecture 0–1",
            "Hafta 3: Systems of equations, echelon form; C basics (hello, loops)",
            "Hafta 4: Inverses, determinants; Python vs C memory model high-level",
            "Hafta 5: Subspaces, column/row space; vectorization mashqlari",
            "Hafta 6: Orthogonality, projections; CS50 pset mini-tasks",
            "Hafta 7: Least squares; from-scratch vector ops (no numpy matmul)",
            "Hafta 8: Consolidation; LA problem set + small write-up",
        ],
    },
    {
        "title": "Oy 5–6: Data structures & Algorithms basics",
        "items": [
            "CS50 davom ettirish + Algorithm practice (LeetCode/Exercism)",
            "CLRS (muhim boblar)",
            "Amaliy: 10 ta algorithm challenge + GitHub",
        ],
        "weeks": [
            "Hafta 1: Big-O, arrays, linked lists; 5 easy LC",
            "Hafta 2: Stacks/queues/hashmaps; 5 easy/medium LC",
            "Hafta 3: Sorting (merge/quick); 3 medium LC (sorting/partition)",
            "Hafta 4: Recursion; CS50 pset on algorithms",
            "Hafta 5: Trees (BST traversals); 4 medium LC",
            "Hafta 6: Graphs (BFS/DFS); 4 medium LC",
            "Hafta 7: DP intro; 3 medium LC (knapsack/coin change)",
            "Hafta 8: Review + 10 challenge push + complexity notes",
        ],
    },
    {
        "title": "Oy 7–9: Statistics & Probability intro",
        "items": [
            "ISLR — statistika tushunchalari",
            "Probability (Khan Academy / MIT OCW)",
            "Amaliy: descriptive stats + hypothesis testing mini-projekt",
        ],
        "weeks": [
            "Hafta 1: Descriptive stats, distributions; pandas describe()",
            "Hafta 2: Sampling, CLT intuitsiyasi; bootstrap demo",
            "Hafta 3: Estimation, confidence intervals; simulyatsiya",
            "Hafta 4: Hypothesis testing (t-test); mini-projekt design",
            "Hafta 5: Multiple testing basics; effect sizes",
            "Hafta 6: Regression intro (ISLR ch.3); sklearn LinearRegression",
            "Hafta 7: Model validation (train/test, CV); metrics RMSE/MAE",
            "Hafta 8–9: Mini-projekt: CSV→EDA→test + report (notebook)",
        ],
    },
    {
        "title": "Oy 10–12: SQL, Linux, practical projects",
        "items": [
            "SQL: SELECT, JOIN, GROUP BY (Postgres)",
            "Linux commandline, Docker basics",
            "Amaliy: CSV -> ETL -> SQL -> BI chart (Power BI/Tableau)",
        ],
        "weeks": [
            "Hafta 1: psql install, SELECT/WHERE; sample DB",
            "Hafta 2: JOINs (INNER/LEFT), GROUP BY/HAVING",
            "Hafta 3: Window functions basics; views",
            "Hafta 4: Linux CLI (grep/sed/awk), bash scripts",
            "Hafta 5: Docker images/containers; docker-compose basics",
            "Hafta 6–7: ETL pipeline: CSV→clean→load Postgres",
            "Hafta 8: BI chart (Tableau/PowerBI) + README + demo",
        ],
    },
]

YEAR2_MONTHLY = [
    {
        "title": "Oy 13–15: Supervised learning + Regression",
        "items": [
            "Linear regression, Ridge/Lasso, model evaluation (RMSE/MAE), CV",
            "Amaliy: Kaggle regression pipeline (feature engineering + model)",
        ],
        "weeks": [
            "Hafta 1: Bias/variance, under/overfitting; baseline model",
            "Hafta 2: Regularization (Ridge/Lasso); scaling, pipelines",
            "Hafta 3: Cross-validation, grid search; metrics deep dive",
            "Hafta 4–5: Feature engineering (dates, text basics); leakage traps",
            "Hafta 6: Kaggle pipeline v1; error analysis",
            "Hafta 7–8: Iterate features + CV; final write-up",
        ],
    },
    {
        "title": "Oy 16–18: Classification + Tree-based models",
        "items": [
            "Logistic regression, Decision Trees, Random Forest, XGBoost",
            "Amaliy: classification project (customer churn va h.k.)",
        ],
        "weeks": [
            "Hafta 1: Logistic regression, thresholds; precision/recall/ROC",
            "Hafta 2: Decision Trees; pruning, depth control",
            "Hafta 3: Random Forest; oob, feature importance",
            "Hafta 4: XGBoost basics; params and early stopping",
            "Hafta 5–6: Churn dataset: EDA→features→models",
            "Hafta 7–8: Calibration, confusion analysis; final report",
        ],
    },
    {
        "title": "Oy 19–20: Econometrics & Causal inference",
        "items": [
            "OLS assumptions, endogeneity, IV (MRU/MIT)",
            "Amaliy: iqtisodiy datasetda OLS + interpretatsiya",
        ],
        "weeks": [
            "Hafta 1: OLS assumptions; residual diagnostics",
            "Hafta 2: Endogeneity, instruments; IV intuition",
            "Hafta 3: DID basics; panel data intro",
            "Hafta 4: Case study: policy dataset; write-up",
        ],
    },
    {
        "title": "Oy 21: A/B testing & Experimentation",
        "items": [
            "Power analysis, sample size, multiple testing correction",
            "Amaliy: A/B test report (real/simulated)",
        ],
        "weeks": [
            "Hafta 1: Experiment design; metrics & guardrails",
            "Hafta 2: Power analysis; sample size calc",
            "Hafta 3: Multiple testing, FDR; sequential pitfalls",
            "Hafta 4: Simulate A/B; analysis notebook + report",
        ],
    },
    {
        "title": "Oy 22–24: Data engineering basics & Deployment",
        "items": [
            "ETL basics, APIs, Docker, basic model deployment (Flask/FastAPI)",
            "Amaliy: REST API orqali model deploy + README + demo",
        ],
        "weeks": [
            "Hafta 1: API design (FastAPI); pydantic schemas",
            "Hafta 2: ETL job (cron/Airflow-lite); tests",
            "Hafta 3: Dockerize API; compose for DB + API",
            "Hafta 4–5: Deploy locally/VM; logging/healthcheck",
            "Hafta 6: Demo + README + curl examples",
        ],
    },
]

YEAR3_MONTHLY = [
    {
        "title": "Oy 25–27: Deep learning basics",
        "items": [
            "Neural networks, backprop, CNNs, RNNs, optimizers",
            "Amaliy: image classification (transfer learning) + Kaggle",
        ],
        "weeks": [
            "Hafta 1: Tensors, autodiff; basic MLP scratch",
            "Hafta 2: Optimizers (SGD/Adam), schedulers; regularization",
            "Hafta 3: CNNs + transfer learning; fine-tune",
            "Hafta 4: Data aug, mixed precision; experiment logging",
            "Hafta 5–6: Kaggle image task end-to-end + report",
        ],
    },
    {
        "title": "Oy 28–30: NLP + Transformers",
        "items": [
            "Tokenization, embeddings, transformers, Hugging Face",
            "Amaliy: text classification/summarization + deploy API",
        ],
        "weeks": [
            "Hafta 1: Tokenizers, subword; embeddings, pooling",
            "Hafta 2: Transformer blocks; finetune pipeline (HF)",
            "Hafta 3: Eval (F1/ROUGE), calibration/bias checks",
            "Hafta 4–5: Build/deploy API; latency basics",
            "Hafta 6: Write-up + demo video",
        ],
    },
    {
        "title": "Oy 31: MLOps & scalable deployment",
        "items": [
            "Model monitoring, CI/CD for ML, containers, cloud basics",
            "Amaliy: deploy with CI + simple monitoring",
        ],
        "weeks": [
            "Hafta 1: CI for notebooks/models; tests & lint",
            "Hafta 2: Container best practices; env separation",
            "Hafta 3: Monitoring metrics & drift; alerts",
            "Hafta 4: Put together demo infra + dashboard",
        ],
    },
    {
        "title": "Oy 32–33: Research & Open Source",
        "items": [
            "Read papers (arXiv), reimplement idea, contribute to OSS",
            "Amaliy: blog/article + GitHub PR",
        ],
        "weeks": [
            "Hafta 1: Paper selection; reading notes",
            "Hafta 2: Reimplementation plan; baseline code",
            "Hafta 3: Experiments; ablations; seeds",
            "Hafta 4: OSS issue selection; small PR",
            "Hafta 5–6: Article polish + submit PR",
        ],
    },
    {
        "title": "Oy 34–36: Job prep, interviews, portfolio",
        "items": [
            "System design for ML, coding interview practice, take-home projects",
            "Amaliy: 3 polished projects + README + video + slides",
        ],
        "weeks": [
            "Hafta 1: Resume/LinkedIn; project selection",
            "Hafta 2: Coding drills (graphs/DP); system design ML",
            "Hafta 3: Take-home dry run; feedback loop",
            "Hafta 4–6: Polish 3 projects; video demos + slides",
        ],
    },
]

PORTFOLIO_ITEMS = [
    "GitHub: 8–12 well-documented repos",
    "Live demos: har project uchun demo yoki video",
    "Technical blog: har loyiha uchun maqola",
    "Certifications: Andrew Ng / fast.ai / CS50",
    "Open-source contributions: 2+ PR",
    "Kaggle: 1–2 kernels/competitions",
    "LinkedIn + CV: linklar va metrics",
]

RESOURCES = [
    ("MIT Linear Algebra (Strang)", "https://ocw.mit.edu"),
    ("Harvard CS50", "https://cs50.harvard.edu"),
    ("Andrew Ng — Machine Learning / deeplearning.ai", "https://www.coursera.org"),
    ("fast.ai — Practical DL", "https://course.fast.ai"),
    ("ISLR — book + labs", "https://www.statlearning.com"),
    ("CLRS — Introduction to Algorithms", "https://mitpress.mit.edu"),
    ("Deep Learning (Goodfellow)", "https://www.deeplearningbook.org"),
    ("Attention Is All You Need", "https://papers.nips.cc"),
    ("PGM — Koller & Friedman", "https://mcb111.org"),
    ("Mostly Harmless Econometrics", "https://www.mostlyharmlesseconometrics.com"),
    ("Apache Airflow docs", "https://airflow.apache.org"),
    ("Hugging Face", "https://huggingface.co"),
    ("Kaggle", "https://www.kaggle.com"),
]

# ----------------------
# Detailed Modules grouped per Year
# ----------------------
MODULES_Y1 = [
    {
        "title": "1) Linear Algebra — PCA/SVD",
        "exercises": [
            "Strang problem sets (har dars 8–12 masala)",
            "From-scratch PCA/SVD (numpy ishlatmasdan)",
        ],
        "deliverable": "linear_algebra_PCA.ipynb",
        "links": [
            ("MIT OCW — Gilbert Strang", "https://ocw.mit.edu"),
            ("MIT 18.06 video lectures", "https://ocw.mit.edu"),
        ],
        "why": "Chiziqli algebra — ML uchun asos",
    },
    {
        "title": "2) Real Analysis & Measure Theory",
        "exercises": [
            "Limit/konvergensiya isbotlari",
            "Lp fazolari bo‘yicha dalillar (weekly proof reading)",
        ],
        "deliverable": "proofs_real_analysis.pdf",
        "links": [
            ("MIT Real Analysis / Measure & Integration", "https://ocw.mit.edu"),
        ],
        "why": "Qat’iy nazariy poydevor",
    },
    {
        "title": "3) Probability & Mathematical Statistics",
        "exercises": [
            "LLN/CLT isbotlari",
            "Monte-Carlo simulyatsiyalar (10 task)",
        ],
        "deliverable": "probability_theory_and_simulations.ipynb",
        "links": [
            ("MIT Probability / Measure & Integration", "https://ocw.mit.edu"),
            ("ISLR", "https://www.statlearning.com"),
        ],
        "why": "Statistik asoslar",
    },
    {
        "title": "4) Algorithms & Data Structures",
        "exercises": [
            "CLRS 40–60 masala",
            "LeetCode 100 (hard/medium): graph/dp/number theory",
        ],
        "deliverable": "algorithms_repo/",
        "links": [
            ("CLRS", "https://mitpress.mit.edu"),
            ("CS50 algorithms", "https://cs50.harvard.edu"),
        ],
        "why": "Algoritmik savodxonlik",
    },
    {
        "title": "5) Programming Foundations (Python/Git/Linux)",
        "exercises": [
            "10 mini-proyekt (data parsing, CLI)",
            "Git branching/PR/workflow",
        ],
        "deliverable": "foundations/ repo (README, tests, CI)",
        "links": [
            ("CS50", "https://cs50.harvard.edu"),
            ("GitHub Learning Lab", "https://lab.github.com"),
        ],
        "why": "Sifatli kod va DevEx",
    },
]

MODULES_Y2 = [
    {
        "title": "6) Statistics & Applied ML (ISLR + CS229)",
        "exercises": [
            "ISLR labs — Python/sklearn",
            "3 ta regression/classification projekti",
        ],
        "deliverable": "islr_replicated.ipynb + blog post",
        "links": [
            ("ISLR", "https://www.statlearning.com"),
            ("CS229 notes", "https://cs229.stanford.edu"),
        ],
        "why": "Nazariya + amaliyot",
    },
    {
        "title": "7) Supervised Learning / Optimization / Convexity",
        "exercises": [
            "Convexity proofs",
            "GD variantlari, SVM solverlarini from-scratch",
        ],
        "deliverable": "optimization_and_solvers.pdf + experiments",
        "links": [
            ("Boyd — Convex Optimization", "https://web.stanford.edu/~boyd/cvxbook/"),
            ("CS229 notes", "https://cs229.stanford.edu"),
        ],
        "why": "Optimallashtirish",
    },
    {
        "title": "8) Tree-based models & Ensembles",
        "exercises": [
            "Bias-variance nazorati (synthetic)",
            "Feature importance stability",
        ],
        "deliverable": "ensemble_study/",
        "links": [
            ("ISLR — Tree methods", "https://www.statlearning.com"),
        ],
        "why": "Amaliy ML",
    },
    {
        "title": "9) Econometrics & Causal Inference",
        "exercises": [
            "IV, DID, panel regressions",
            "Sensitivity analyses",
        ],
        "deliverable": "Causal notebook + write-up",
        "links": [
            ("Mostly Harmless Econometrics", "https://www.mostlyharmlesseconometrics.com"),
        ],
        "why": "Sababiy tahlil",
    },
    {
        "title": "10) A/B Testing & Experimentation",
        "exercises": [
            "Power analysis, sequential testing",
            "FDR control simulyatsiyalari",
        ],
        "deliverable": "AB_experiment_pipeline.ipynb + dashboard mockup",
        "links": [],
        "why": "Eksperimentlar dizayni",
    },
    {
        "title": "11) Data Engineering & Reproducible Research",
        "exercises": [
            "Airflow ETL",
            "DVC data versioning + CI tests",
        ],
        "deliverable": "Dockerized ETL + reproducibility README",
        "links": [
            ("Apache Airflow", "https://airflow.apache.org"),
            ("DVC", "https://dvc.org"),
        ],
        "why": "Ishlab chiqarishga tayyor quvur",
    },
]

MODULES_Y3 = [
    {
        "title": "12) Deep Learning (foundations → advanced)",
        "exercises": [
            "From-scratch NN",
            "CNN transfer-learning project",
        ],
        "deliverable": "dl_foundations.ipynb + video",
        "links": [
            ("Deep Learning (Goodfellow)", "https://www.deeplearningbook.org"),
            ("fast.ai Practical DL", "https://course.fast.ai"),
        ],
        "why": "DL poydevori",
    },
    {
        "title": "13) NLP & Transformers",
        "exercises": [
            "Domain datasetga fine-tune",
            "Calibration & bias evaluation",
        ],
        "deliverable": "Deployed API + bias write-up",
        "links": [
            ("Attention Is All You Need", "https://papers.nips.cc"),
            ("Hugging Face", "https://huggingface.co"),
        ],
        "why": "Zamonaviy NLP",
    },
    {
        "title": "14) Probabilistic Graphical Models & Bayesian",
        "exercises": [
            "Variational inference / MCMC",
            "Diagnostics",
        ],
        "deliverable": "pgm_bayesian.ipynb + technical note",
        "links": [],
        "why": "Noaniqlikni modellashtirish",
    },
    {
        "title": "15) Theory of Deep Learning & Reproducibility",
        "exercises": [
            "PAC-Bayes / generalization tajribasi",
            "Qisqa ilmiy yozuv",
        ],
        "deliverable": "Reproducible repo + 3–6 sahifa paper",
        "links": [],
        "why": "Nazariy chuqurlik",
    },
    {
        "title": "16) MLOps, Production & Monitoring",
        "exercises": [
            "Serve model (Docker/K8s)",
            "Data drift & metrics monitoring",
        ],
        "deliverable": "Production repo + runbook + dashboard",
        "links": [],
        "why": "Skalalanadigan ML",
    },
    {
        "title": "17) Capstone / Thesis-like Projects",
        "exercises": [
            "3 publication-quality projects",
            "Slides + video demos",
        ],
        "deliverable": "Capstone repo (papers, notebooks, deployment)",
        "links": [],
        "why": "Ishga tayyor portfolio",
    },
]


# ----------------------
# Defaults for planners
# ----------------------
DAYS_7 = [
    "Kun 1", "Kun 2", "Kun 3", "Kun 4", "Kun 5", "Kun 6", "Kun 7"
]


def default_sprint_tasks() -> Dict[str, Dict[str, str]]:
    template = {
        "Kun 1": "CS50 - Lecture 0 tomosha + GitHub repo",
        "Kun 2": "Python basics (functions, lists, dicts) + mini mashqlar",
        "Kun 3": "Linear Algebra — vectors (MIT OCW)",
        "Kun 4": "SQL — SELECT, JOIN amaliyoti",
        "Kun 5": "ISLR — Chapter 1 + small lab",
        "Kun 6": "Andrew Ng — ML Lecture 1",
        "Kun 7": "Mini loyiha: EDA on small dataset + push",
    }
    return {
        day: {"task": template[day], "done": False, "notes": ""}
        for day in DAYS_7
    }


WEEK_DAYS = [
    "Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"
]


def default_weekly_template() -> Dict[str, List[Dict[str, str]]]:
    return {
        "Dushanba": [
            {"slot": "3 soat kurs", "topic": ""},
            {"slot": "1 soat leetcode", "topic": ""},
        ],
        "Seshanba": [
            {"slot": "3 soat matematika/LA", "topic": ""},
            {"slot": "1 soat notes", "topic": ""},
        ],
        "Chorshanba": [
            {"slot": "3 soat coding (project)", "topic": ""},
        ],
        "Payshanba": [
            {"slot": "3 soat stats/ML nazariya", "topic": ""},
        ],
        "Juma": [
            {"slot": "2 soat deployment/engineering", "topic": ""},
        ],
        "Shanba": [
            {"slot": "4–6 soat hack/OSS/Kaggle", "topic": ""},
        ],
        "Yakshanba": [
            {"slot": "2–3 soat review + blog", "topic": ""},
        ],
    }


# ----------------------
# UI Components
# ----------------------

def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def checklist(block_key: str, items: List[str], state_key: str) -> None:
    state: Dict[str, bool] = st.session_state[state_key]
    cols = st.columns(2)
    for idx, item in enumerate(items):
        col = cols[idx % 2]
        with col:
            checked = state.get(f"{block_key}:{item}", False)
            new_val = st.checkbox(item, value=checked, key=f"cb:{block_key}:{idx}")
            if new_val != checked:
                state[f"{block_key}:{item}"] = new_val


def monthly_sections(title: str, monthly: List[Dict[str, List[str]]], state_key: str) -> None:
    st.markdown(f"### {title}")
    for i, month in enumerate(monthly, start=1):
        with st.expander(month["title"], expanded=False):
            checklist(f"{title}-{i}", month["items"], state_key)
            weeks = month.get("weeks", [])
            if weeks:
                st.markdown("**Haftalik reja (oddiy matn):**")
                st.markdown("\n".join([f"- {w}" for w in weeks]))


def module_sections(title: str, modules_list: List[Dict[str, List[str]]], state_key: str, prefix: str) -> None:
    st.markdown(f"### {title}")
    for i, mod in enumerate(modules_list, start=1):
        with st.expander(mod["title"], expanded=False):
            if mod.get("why"):
                st.caption(mod["why"])
            if mod.get("links"):
                for (name, url) in mod["links"]:
                    st.markdown(f"- [{name}]({url})")
            st.markdown("**Mashqlar**")
            checklist(f"{prefix}-mod-{i}", mod["exercises"], state_key)
            st.markdown(f"**Deliverable**: `{mod['deliverable']}`")
            st.text_area("Notes", key=f"notes_{prefix}_{i}")


def kpi_progress(state_key: str, title: str) -> None:
    state: Dict[str, bool] = st.session_state[state_key]
    total = len(state)
    done = sum(1 for v in state.values() if v)
    pct = (done / total) * 100 if total else 0
    st.metric(label=title, value=f"{done}/{total}", delta=f"{pct:.0f}%")


# ----------------------
# Pages
# ----------------------

def page_overview() -> None:
    section_header("PlanA — 3 Yillik O‘quv Reja", "Asoslar → Data Science → Advanced ML/AI")
    st.write(
        "Bu platforma uch yillik rejani boshqarish, haftalik reja va 7 kunlik sprintlarni rejalashtirish, "
        "hamda resurslar va portfolioni kuzatish uchun yaratilgan."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_progress(PROGRESS_KEYS["y1"], "Yil 1 Progress")
    with col2:
        kpi_progress(PROGRESS_KEYS["y2"], "Yil 2 Progress")
    with col3:
        kpi_progress(PROGRESS_KEYS["y3"], "Yil 3 Progress")

    st.divider()
    st.markdown("### Asosiy manbalar (load-bearing)")
    st.markdown("- MIT Linear Algebra (Strang)\n- Harvard CS50\n- Andrew Ng ML / deeplearning.ai\n- fast.ai Practical DL\n- ISLR (book + labs)")


def page_year1() -> None:
    section_header("Yil 1 — Asoslar", "Matematika, Python, CS, Linux/Git, SQL")
    monthly_sections("Yil 1", YEAR1_MONTHLY, PROGRESS_KEYS["y1"])
    module_sections("Yil 1 — Tafsilotli Modullar", MODULES_Y1, PROGRESS_KEYS["y1"], prefix="y1")


def page_year2() -> None:
    section_header("Yil 2 — Data Science & Applied Economics", "Statistika, ML, Econometrics, A/B, Deploy")
    monthly_sections("Yil 2", YEAR2_MONTHLY, PROGRESS_KEYS["y2"])
    module_sections("Yil 2 — Tafsilotli Modullar", MODULES_Y2, PROGRESS_KEYS["y2"], prefix="y2")


def page_year3() -> None:
    section_header("Yil 3 — Advanced ML / AI & Job Prep", "DL, NLP, MLOps, Research, Interviews")
    monthly_sections("Yil 3", YEAR3_MONTHLY, PROGRESS_KEYS["y3"])
    module_sections("Yil 3 — Tafsilotli Modullar", MODULES_Y3, PROGRESS_KEYS["y3"], prefix="y3")


def page_sprint_7d() -> None:
    section_header("7 kunlik Sprint", "Darhol boshlash uchun konkret reja")
    data = st.session_state[PROGRESS_KEYS["sprint"]]
    for day in DAYS_7:
        with st.expander(day, expanded=False):
            data[day]["task"] = st.text_input("Vazifa", value=data[day]["task"], key=f"sp_task_{day}")
            data[day]["notes"] = st.text_area("Izohlar", value=data[day]["notes"], key=f"sp_notes_{day}")
            data[day]["done"] = st.checkbox("Bajarildi", value=data[day]["done"], key=f"sp_done_{day}")
    done = sum(1 for d in data.values() if d["done"])
    st.success(f"Yakun: {done}/7 kun bajarildi")


def page_weekly_plan() -> None:
    section_header("Haftalik o‘qish jadvali", "Ish/ta’limga mos fleksibl shablon")
    plan = st.session_state[PROGRESS_KEYS["weekly"]]
    for day in WEEK_DAYS:
        with st.expander(day, expanded=False):
            blocks = plan[day]
            for i, block in enumerate(blocks):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.text_input("Vaqt/Slot", value=block["slot"], key=f"wk_slot_{day}_{i}")
                with cols[1]:
                    block["topic"] = st.text_input("Mavzu", value=block["topic"], key=f"wk_topic_{day}_{i}")

    st.info("Maslahat: Google Calendar/Notion bilan sinxronlashtiring.")


def page_portfolio_resources() -> None:
    section_header("Portfolio & Resurslar")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Portfolio Checklist")
        checklist("portfolio", PORTFOLIO_ITEMS, PROGRESS_KEYS["portfolio"])

    with col2:
        st.markdown("### Resource Hub")
        res_state: Dict[str, bool] = st.session_state[PROGRESS_KEYS["resources"]]
        for i, (name, url) in enumerate(RESOURCES):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"- [{name}]({url})")
            with cols[1]:
                key = f"res:{name}"
                val = res_state.get(key, False)
                res_state[key] = st.checkbox("Seen", value=val, key=f"res_seen_{i}")

    st.divider()
    st.markdown("### Majburiy amaliy loyihalar (kamida 8 ta)")
    must_projects = [
        "Data cleaning & EDA — blog post + Kaggle dataset",
        "Regression — price prediction + explainability",
        "Classification — churn/health + metrics report",
        "Time-series — ARIMA/Prophet/NN",
        "NLP — sentiment/summarization (deploy)",
        "CV — image classify + transfer learning",
        "End-to-end deployed ML API — frontend demo",
        "Open-source contribution yoki reproduced paper",
    ]
    checklist("projects", must_projects, PROGRESS_KEYS["portfolio"])


# ----------------------
# Main
# ----------------------

def main() -> None:
    st.set_page_config(page_title="PlanA — 3-Year ML/AI", layout="wide")
    ensure_session_state_defaults()

    with st.sidebar:
        st.markdown("### Navigatsiya")
        page = st.radio(
            "Bo‘limni tanlang",
            (
                "Overview",
                "Yil 1",
                "Yil 2",
                "Yil 3",
                "7 kunlik Sprint",
                "Haftalik Reja",
                "Portfolio & Resurslar",
            ),
            index=0,
        )
        st.divider()
        st.caption("Session-based progress. Persistent storage emas (hozircha).")

    if page == "Overview":
        page_overview()
    elif page == "Yil 1":
        page_year1()
    elif page == "Yil 2":
        page_year2()
    elif page == "Yil 3":
        page_year3()
    elif page == "7 kunlik Sprint":
        page_sprint_7d()
    elif page == "Haftalik Reja":
        page_weekly_plan()
    elif page == "Portfolio & Resurslar":
        page_portfolio_resources()


if __name__ == "__main__":
    main()
