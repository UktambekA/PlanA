# import datetime as dt
# from dateutil.relativedelta import relativedelta
# from typing import Dict, List

# import pandas as pd
# import streamlit as st


# # ----------------------
# # Session State Helpers
# # ----------------------
# PROGRESS_KEYS = {
#     "y1": "progress_year1",
#     "y2": "progress_year2",
#     "y3": "progress_year3",
#     "portfolio": "progress_portfolio",
#     "resources": "progress_resources",
#     "sprint": "sprint_tasks",
#     "weekly": "weekly_plan",
# }


# def ensure_session_state_defaults() -> None:
#     if PROGRESS_KEYS["y1"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["y1"]] = {}
#     if PROGRESS_KEYS["y2"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["y2"]] = {}
#     if PROGRESS_KEYS["y3"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["y3"]] = {}
#     if PROGRESS_KEYS["portfolio"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["portfolio"]] = {}
#     if PROGRESS_KEYS["resources"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["resources"]] = {}
#     if PROGRESS_KEYS["sprint"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["sprint"]] = default_sprint_tasks()
#     if PROGRESS_KEYS["weekly"] not in st.session_state:
#         st.session_state[PROGRESS_KEYS["weekly"]] = default_weekly_template()


# # ----------------------
# # Data Definitions
# # ----------------------
# YEAR1_MONTHLY = [
#     {
#         "title": "Oy 1–2: Matematika & Python basics",
#         "items": [
#             "Math: Precalculus / Calculus I (Khan Academy yoki MIT OCW)",
#             "Python: sintaksis, funksiyalar, list/dict, virtualenv",
#             "Amaliy: 10 mini kod mashqlari + GitHub repo",
#         ],
#         "weeks": [
#             "Hafta 1: Precalculus reviziya; Python sintaksis, types, input/output",
#             "Hafta 2: Calculus I — limits/derivatives; Python functions & modules",
#             "Hafta 3: Derivatives qo‘llanmalar; data structures (list/dict/set)",
#             "Hafta 4: Integrals kirish; venv, pip, basic file I/O + 3 mini-mashq",
#             "Hafta 5: Applications of integrals; exceptions, logging + 3 mini-mashq",
#             "Hafta 6: Series/approx; CLI tools, argparse, notebooks + 4 mini-mashq",
#             "Hafta 7: Git/GitHub: repo, branching, PR flow; first README",
#             "Hafta 8: Repeat & consolidation; small EDA on CSV (pandas kirish)",
#         ],
#     },
#     {
#         "title": "Oy 3–4: Linear Algebra + CS50 start",
#         "items": [
#             "Linear Algebra (MIT OCW, Gilbert Strang)",
#             "CS50 boshlash — algoritmlar, C va Python",
#             "Amaliy: vektor/matrix kodlash, notebooklar",
#         ],
#         "weeks": [
#             "Hafta 1: Vectors, linear combinations; NumPy basics (faqat tekshirish)",
#             "Hafta 2: Matrices, matrix multiplication; CS50 Lecture 0–1",
#             "Hafta 3: Systems of equations, echelon form; C basics (hello, loops)",
#             "Hafta 4: Inverses, determinants; Python vs C memory model high-level",
#             "Hafta 5: Subspaces, column/row space; vectorization mashqlari",
#             "Hafta 6: Orthogonality, projections; CS50 pset mini-tasks",
#             "Hafta 7: Least squares; from-scratch vector ops (no numpy matmul)",
#             "Hafta 8: Consolidation; LA problem set + small write-up",
#         ],
#     },
#     {
#         "title": "Oy 5–6: Data structures & Algorithms basics",
#         "items": [
#             "CS50 davom ettirish + Algorithm practice (LeetCode/Exercism)",
#             "CLRS (muhim boblar)",
#             "Amaliy: 10 ta algorithm challenge + GitHub",
#         ],
#         "weeks": [
#             "Hafta 1: Big-O, arrays, linked lists; 5 easy LC",
#             "Hafta 2: Stacks/queues/hashmaps; 5 easy/medium LC",
#             "Hafta 3: Sorting (merge/quick); 3 medium LC (sorting/partition)",
#             "Hafta 4: Recursion; CS50 pset on algorithms",
#             "Hafta 5: Trees (BST traversals); 4 medium LC",
#             "Hafta 6: Graphs (BFS/DFS); 4 medium LC",
#             "Hafta 7: DP intro; 3 medium LC (knapsack/coin change)",
#             "Hafta 8: Review + 10 challenge push + complexity notes",
#         ],
#     },
#     {
#         "title": "Oy 7–9: Statistics & Probability intro",
#         "items": [
#             "ISLR — statistika tushunchalari",
#             "Probability (Khan Academy / MIT OCW)",
#             "Amaliy: descriptive stats + hypothesis testing mini-projekt",
#         ],
#         "weeks": [
#             "Hafta 1: Descriptive stats, distributions; pandas describe()",
#             "Hafta 2: Sampling, CLT intuitsiyasi; bootstrap demo",
#             "Hafta 3: Estimation, confidence intervals; simulyatsiya",
#             "Hafta 4: Hypothesis testing (t-test); mini-projekt design",
#             "Hafta 5: Multiple testing basics; effect sizes",
#             "Hafta 6: Regression intro (ISLR ch.3); sklearn LinearRegression",
#             "Hafta 7: Model validation (train/test, CV); metrics RMSE/MAE",
#             "Hafta 8–9: Mini-projekt: CSV→EDA→test + report (notebook)",
#         ],
#     },
#     {
#         "title": "Oy 10–12: SQL, Linux, practical projects",
#         "items": [
#             "SQL: SELECT, JOIN, GROUP BY (Postgres)",
#             "Linux commandline, Docker basics",
#             "Amaliy: CSV -> ETL -> SQL -> BI chart (Power BI/Tableau)",
#         ],
#         "weeks": [
#             "Hafta 1: psql install, SELECT/WHERE; sample DB",
#             "Hafta 2: JOINs (INNER/LEFT), GROUP BY/HAVING",
#             "Hafta 3: Window functions basics; views",
#             "Hafta 4: Linux CLI (grep/sed/awk), bash scripts",
#             "Hafta 5: Docker images/containers; docker-compose basics",
#             "Hafta 6–7: ETL pipeline: CSV→clean→load Postgres",
#             "Hafta 8: BI chart (Tableau/PowerBI) + README + demo",
#         ],
#     },
# ]

# YEAR2_MONTHLY = [
#     {
#         "title": "Oy 13–15: Supervised learning + Regression",
#         "items": [
#             "Linear regression, Ridge/Lasso, model evaluation (RMSE/MAE), CV",
#             "Amaliy: Kaggle regression pipeline (feature engineering + model)",
#         ],
#         "weeks": [
#             "Hafta 1: Bias/variance, under/overfitting; baseline model",
#             "Hafta 2: Regularization (Ridge/Lasso); scaling, pipelines",
#             "Hafta 3: Cross-validation, grid search; metrics deep dive",
#             "Hafta 4–5: Feature engineering (dates, text basics); leakage traps",
#             "Hafta 6: Kaggle pipeline v1; error analysis",
#             "Hafta 7–8: Iterate features + CV; final write-up",
#         ],
#     },
#     {
#         "title": "Oy 16–18: Classification + Tree-based models",
#         "items": [
#             "Logistic regression, Decision Trees, Random Forest, XGBoost",
#             "Amaliy: classification project (customer churn va h.k.)",
#         ],
#         "weeks": [
#             "Hafta 1: Logistic regression, thresholds; precision/recall/ROC",
#             "Hafta 2: Decision Trees; pruning, depth control",
#             "Hafta 3: Random Forest; oob, feature importance",
#             "Hafta 4: XGBoost basics; params and early stopping",
#             "Hafta 5–6: Churn dataset: EDA→features→models",
#             "Hafta 7–8: Calibration, confusion analysis; final report",
#         ],
#     },
#     {
#         "title": "Oy 19–20: Econometrics & Causal inference",
#         "items": [
#             "OLS assumptions, endogeneity, IV (MRU/MIT)",
#             "Amaliy: iqtisodiy datasetda OLS + interpretatsiya",
#         ],
#         "weeks": [
#             "Hafta 1: OLS assumptions; residual diagnostics",
#             "Hafta 2: Endogeneity, instruments; IV intuition",
#             "Hafta 3: DID basics; panel data intro",
#             "Hafta 4: Case study: policy dataset; write-up",
#         ],
#     },
#     {
#         "title": "Oy 21: A/B testing & Experimentation",
#         "items": [
#             "Power analysis, sample size, multiple testing correction",
#             "Amaliy: A/B test report (real/simulated)",
#         ],
#         "weeks": [
#             "Hafta 1: Experiment design; metrics & guardrails",
#             "Hafta 2: Power analysis; sample size calc",
#             "Hafta 3: Multiple testing, FDR; sequential pitfalls",
#             "Hafta 4: Simulate A/B; analysis notebook + report",
#         ],
#     },
#     {
#         "title": "Oy 22–24: Data engineering basics & Deployment",
#         "items": [
#             "ETL basics, APIs, Docker, basic model deployment (Flask/FastAPI)",
#             "Amaliy: REST API orqali model deploy + README + demo",
#         ],
#         "weeks": [
#             "Hafta 1: API design (FastAPI); pydantic schemas",
#             "Hafta 2: ETL job (cron/Airflow-lite); tests",
#             "Hafta 3: Dockerize API; compose for DB + API",
#             "Hafta 4–5: Deploy locally/VM; logging/healthcheck",
#             "Hafta 6: Demo + README + curl examples",
#         ],
#     },
# ]

# YEAR3_MONTHLY = [
#     {
#         "title": "Oy 25–27: Deep learning basics",
#         "items": [
#             "Neural networks, backprop, CNNs, RNNs, optimizers",
#             "Amaliy: image classification (transfer learning) + Kaggle",
#         ],
#         "weeks": [
#             "Hafta 1: Tensors, autodiff; basic MLP scratch",
#             "Hafta 2: Optimizers (SGD/Adam), schedulers; regularization",
#             "Hafta 3: CNNs + transfer learning; fine-tune",
#             "Hafta 4: Data aug, mixed precision; experiment logging",
#             "Hafta 5–6: Kaggle image task end-to-end + report",
#         ],
#     },
#     {
#         "title": "Oy 28–30: NLP + Transformers",
#         "items": [
#             "Tokenization, embeddings, transformers, Hugging Face",
#             "Amaliy: text classification/summarization + deploy API",
#         ],
#         "weeks": [
#             "Hafta 1: Tokenizers, subword; embeddings, pooling",
#             "Hafta 2: Transformer blocks; finetune pipeline (HF)",
#             "Hafta 3: Eval (F1/ROUGE), calibration/bias checks",
#             "Hafta 4–5: Build/deploy API; latency basics",
#             "Hafta 6: Write-up + demo video",
#         ],
#     },
#     {
#         "title": "Oy 31: MLOps & scalable deployment",
#         "items": [
#             "Model monitoring, CI/CD for ML, containers, cloud basics",
#             "Amaliy: deploy with CI + simple monitoring",
#         ],
#         "weeks": [
#             "Hafta 1: CI for notebooks/models; tests & lint",
#             "Hafta 2: Container best practices; env separation",
#             "Hafta 3: Monitoring metrics & drift; alerts",
#             "Hafta 4: Put together demo infra + dashboard",
#         ],
#     },
#     {
#         "title": "Oy 32–33: Research & Open Source",
#         "items": [
#             "Read papers (arXiv), reimplement idea, contribute to OSS",
#             "Amaliy: blog/article + GitHub PR",
#         ],
#         "weeks": [
#             "Hafta 1: Paper selection; reading notes",
#             "Hafta 2: Reimplementation plan; baseline code",
#             "Hafta 3: Experiments; ablations; seeds",
#             "Hafta 4: OSS issue selection; small PR",
#             "Hafta 5–6: Article polish + submit PR",
#         ],
#     },
#     {
#         "title": "Oy 34–36: Job prep, interviews, portfolio",
#         "items": [
#             "System design for ML, coding interview practice, take-home projects",
#             "Amaliy: 3 polished projects + README + video + slides",
#         ],
#         "weeks": [
#             "Hafta 1: Resume/LinkedIn; project selection",
#             "Hafta 2: Coding drills (graphs/DP); system design ML",
#             "Hafta 3: Take-home dry run; feedback loop",
#             "Hafta 4–6: Polish 3 projects; video demos + slides",
#         ],
#     },
# ]

# PORTFOLIO_ITEMS = [
#     "GitHub: 8–12 well-documented repos",
#     "Live demos: har project uchun demo yoki video",
#     "Technical blog: har loyiha uchun maqola",
#     "Certifications: Andrew Ng / fast.ai / CS50",
#     "Open-source contributions: 2+ PR",
#     "Kaggle: 1–2 kernels/competitions",
#     "LinkedIn + CV: linklar va metrics",
# ]

# RESOURCES = [
#     ("MIT Linear Algebra (Strang)", "https://ocw.mit.edu"),
#     ("Harvard CS50", "https://cs50.harvard.edu"),
#     ("Andrew Ng — Machine Learning / deeplearning.ai", "https://www.coursera.org"),
#     ("fast.ai — Practical DL", "https://course.fast.ai"),
#     ("ISLR — book + labs", "https://www.statlearning.com"),
#     ("CLRS — Introduction to Algorithms", "https://mitpress.mit.edu"),
#     ("Deep Learning (Goodfellow)", "https://www.deeplearningbook.org"),
#     ("Attention Is All You Need", "https://papers.nips.cc"),
#     ("PGM — Koller & Friedman", "https://mcb111.org"),
#     ("Mostly Harmless Econometrics", "https://www.mostlyharmlesseconometrics.com"),
#     ("Apache Airflow docs", "https://airflow.apache.org"),
#     ("Hugging Face", "https://huggingface.co"),
#     ("Kaggle", "https://www.kaggle.com"),
# ]

# # ----------------------
# # Detailed Modules grouped per Year
# # ----------------------
# MODULES_Y1 = [
#     {
#         "title": "1) Linear Algebra — PCA/SVD",
#         "exercises": [
#             "Strang problem sets (har dars 8–12 masala)",
#             "From-scratch PCA/SVD (numpy ishlatmasdan)",
#         ],
#         "deliverable": "linear_algebra_PCA.ipynb",
#         "links": [
#             ("MIT OCW — Gilbert Strang", "https://ocw.mit.edu"),
#             ("MIT 18.06 video lectures", "https://ocw.mit.edu"),
#         ],
#         "why": "Chiziqli algebra — ML uchun asos",
#     },
#     {
#         "title": "2) Real Analysis & Measure Theory",
#         "exercises": [
#             "Limit/konvergensiya isbotlari",
#             "Lp fazolari bo‘yicha dalillar (weekly proof reading)",
#         ],
#         "deliverable": "proofs_real_analysis.pdf",
#         "links": [
#             ("MIT Real Analysis / Measure & Integration", "https://ocw.mit.edu"),
#         ],
#         "why": "Qat’iy nazariy poydevor",
#     },
#     {
#         "title": "3) Probability & Mathematical Statistics",
#         "exercises": [
#             "LLN/CLT isbotlari",
#             "Monte-Carlo simulyatsiyalar (10 task)",
#         ],
#         "deliverable": "probability_theory_and_simulations.ipynb",
#         "links": [
#             ("MIT Probability / Measure & Integration", "https://ocw.mit.edu"),
#             ("ISLR", "https://www.statlearning.com"),
#         ],
#         "why": "Statistik asoslar",
#     },
#     {
#         "title": "4) Algorithms & Data Structures",
#         "exercises": [
#             "CLRS 40–60 masala",
#             "LeetCode 100 (hard/medium): graph/dp/number theory",
#         ],
#         "deliverable": "algorithms_repo/",
#         "links": [
#             ("CLRS", "https://mitpress.mit.edu"),
#             ("CS50 algorithms", "https://cs50.harvard.edu"),
#         ],
#         "why": "Algoritmik savodxonlik",
#     },
#     {
#         "title": "5) Programming Foundations (Python/Git/Linux)",
#         "exercises": [
#             "10 mini-proyekt (data parsing, CLI)",
#             "Git branching/PR/workflow",
#         ],
#         "deliverable": "foundations/ repo (README, tests, CI)",
#         "links": [
#             ("CS50", "https://cs50.harvard.edu"),
#             ("GitHub Learning Lab", "https://lab.github.com"),
#         ],
#         "why": "Sifatli kod va DevEx",
#     },
# ]

# MODULES_Y2 = [
#     {
#         "title": "6) Statistics & Applied ML (ISLR + CS229)",
#         "exercises": [
#             "ISLR labs — Python/sklearn",
#             "3 ta regression/classification projekti",
#         ],
#         "deliverable": "islr_replicated.ipynb + blog post",
#         "links": [
#             ("ISLR", "https://www.statlearning.com"),
#             ("CS229 notes", "https://cs229.stanford.edu"),
#         ],
#         "why": "Nazariya + amaliyot",
#     },
#     {
#         "title": "7) Supervised Learning / Optimization / Convexity",
#         "exercises": [
#             "Convexity proofs",
#             "GD variantlari, SVM solverlarini from-scratch",
#         ],
#         "deliverable": "optimization_and_solvers.pdf + experiments",
#         "links": [
#             ("Boyd — Convex Optimization", "https://web.stanford.edu/~boyd/cvxbook/"),
#             ("CS229 notes", "https://cs229.stanford.edu"),
#         ],
#         "why": "Optimallashtirish",
#     },
#     {
#         "title": "8) Tree-based models & Ensembles",
#         "exercises": [
#             "Bias-variance nazorati (synthetic)",
#             "Feature importance stability",
#         ],
#         "deliverable": "ensemble_study/",
#         "links": [
#             ("ISLR — Tree methods", "https://www.statlearning.com"),
#         ],
#         "why": "Amaliy ML",
#     },
#     {
#         "title": "9) Econometrics & Causal Inference",
#         "exercises": [
#             "IV, DID, panel regressions",
#             "Sensitivity analyses",
#         ],
#         "deliverable": "Causal notebook + write-up",
#         "links": [
#             ("Mostly Harmless Econometrics", "https://www.mostlyharmlesseconometrics.com"),
#         ],
#         "why": "Sababiy tahlil",
#     },
#     {
#         "title": "10) A/B Testing & Experimentation",
#         "exercises": [
#             "Power analysis, sequential testing",
#             "FDR control simulyatsiyalari",
#         ],
#         "deliverable": "AB_experiment_pipeline.ipynb + dashboard mockup",
#         "links": [],
#         "why": "Eksperimentlar dizayni",
#     },
#     {
#         "title": "11) Data Engineering & Reproducible Research",
#         "exercises": [
#             "Airflow ETL",
#             "DVC data versioning + CI tests",
#         ],
#         "deliverable": "Dockerized ETL + reproducibility README",
#         "links": [
#             ("Apache Airflow", "https://airflow.apache.org"),
#             ("DVC", "https://dvc.org"),
#         ],
#         "why": "Ishlab chiqarishga tayyor quvur",
#     },
# ]

# MODULES_Y3 = [
#     {
#         "title": "12) Deep Learning (foundations → advanced)",
#         "exercises": [
#             "From-scratch NN",
#             "CNN transfer-learning project",
#         ],
#         "deliverable": "dl_foundations.ipynb + video",
#         "links": [
#             ("Deep Learning (Goodfellow)", "https://www.deeplearningbook.org"),
#             ("fast.ai Practical DL", "https://course.fast.ai"),
#         ],
#         "why": "DL poydevori",
#     },
#     {
#         "title": "13) NLP & Transformers",
#         "exercises": [
#             "Domain datasetga fine-tune",
#             "Calibration & bias evaluation",
#         ],
#         "deliverable": "Deployed API + bias write-up",
#         "links": [
#             ("Attention Is All You Need", "https://papers.nips.cc"),
#             ("Hugging Face", "https://huggingface.co"),
#         ],
#         "why": "Zamonaviy NLP",
#     },
#     {
#         "title": "14) Probabilistic Graphical Models & Bayesian",
#         "exercises": [
#             "Variational inference / MCMC",
#             "Diagnostics",
#         ],
#         "deliverable": "pgm_bayesian.ipynb + technical note",
#         "links": [],
#         "why": "Noaniqlikni modellashtirish",
#     },
#     {
#         "title": "15) Theory of Deep Learning & Reproducibility",
#         "exercises": [
#             "PAC-Bayes / generalization tajribasi",
#             "Qisqa ilmiy yozuv",
#         ],
#         "deliverable": "Reproducible repo + 3–6 sahifa paper",
#         "links": [],
#         "why": "Nazariy chuqurlik",
#     },
#     {
#         "title": "16) MLOps, Production & Monitoring",
#         "exercises": [
#             "Serve model (Docker/K8s)",
#             "Data drift & metrics monitoring",
#         ],
#         "deliverable": "Production repo + runbook + dashboard",
#         "links": [],
#         "why": "Skalalanadigan ML",
#     },
#     {
#         "title": "17) Capstone / Thesis-like Projects",
#         "exercises": [
#             "3 publication-quality projects",
#             "Slides + video demos",
#         ],
#         "deliverable": "Capstone repo (papers, notebooks, deployment)",
#         "links": [],
#         "why": "Ishga tayyor portfolio",
#     },
# ]


# # ----------------------
# # Defaults for planners
# # ----------------------
# DAYS_7 = [
#     "Kun 1", "Kun 2", "Kun 3", "Kun 4", "Kun 5", "Kun 6", "Kun 7"
# ]


# def default_sprint_tasks() -> Dict[str, Dict[str, str]]:
#     template = {
#         "Kun 1": "CS50 - Lecture 0 tomosha + GitHub repo",
#         "Kun 2": "Python basics (functions, lists, dicts) + mini mashqlar",
#         "Kun 3": "Linear Algebra — vectors (MIT OCW)",
#         "Kun 4": "SQL — SELECT, JOIN amaliyoti",
#         "Kun 5": "ISLR — Chapter 1 + small lab",
#         "Kun 6": "Andrew Ng — ML Lecture 1",
#         "Kun 7": "Mini loyiha: EDA on small dataset + push",
#     }
#     return {
#         day: {"task": template[day], "done": False, "notes": ""}
#         for day in DAYS_7
#     }


# WEEK_DAYS = [
#     "Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"
# ]


# def default_weekly_template() -> Dict[str, List[Dict[str, str]]]:
#     return {
#         "Dushanba": [
#             {"slot": "3 soat kurs", "topic": ""},
#             {"slot": "1 soat leetcode", "topic": ""},
#         ],
#         "Seshanba": [
#             {"slot": "3 soat matematika/LA", "topic": ""},
#             {"slot": "1 soat notes", "topic": ""},
#         ],
#         "Chorshanba": [
#             {"slot": "3 soat coding (project)", "topic": ""},
#         ],
#         "Payshanba": [
#             {"slot": "3 soat stats/ML nazariya", "topic": ""},
#         ],
#         "Juma": [
#             {"slot": "2 soat deployment/engineering", "topic": ""},
#         ],
#         "Shanba": [
#             {"slot": "4–6 soat hack/OSS/Kaggle", "topic": ""},
#         ],
#         "Yakshanba": [
#             {"slot": "2–3 soat review + blog", "topic": ""},
#         ],
#     }


# # ----------------------
# # UI Components
# # ----------------------

# def section_header(title: str, subtitle: str = "") -> None:
#     st.markdown(f"## {title}")
#     if subtitle:
#         st.caption(subtitle)


# def checklist(block_key: str, items: List[str], state_key: str) -> None:
#     state: Dict[str, bool] = st.session_state[state_key]
#     cols = st.columns(2)
#     for idx, item in enumerate(items):
#         col = cols[idx % 2]
#         with col:
#             checked = state.get(f"{block_key}:{item}", False)
#             new_val = st.checkbox(item, value=checked, key=f"cb:{block_key}:{idx}")
#             if new_val != checked:
#                 state[f"{block_key}:{item}"] = new_val


# def monthly_sections(title: str, monthly: List[Dict[str, List[str]]], state_key: str) -> None:
#     st.markdown(f"### {title}")
#     for i, month in enumerate(monthly, start=1):
#         with st.expander(month["title"], expanded=False):
#             checklist(f"{title}-{i}", month["items"], state_key)
#             weeks = month.get("weeks", [])
#             if weeks:
#                 st.markdown("**Haftalik reja (oddiy matn):**")
#                 st.markdown("\n".join([f"- {w}" for w in weeks]))


# def module_sections(title: str, modules_list: List[Dict[str, List[str]]], state_key: str, prefix: str) -> None:
#     st.markdown(f"### {title}")
#     for i, mod in enumerate(modules_list, start=1):
#         with st.expander(mod["title"], expanded=False):
#             if mod.get("why"):
#                 st.caption(mod["why"])
#             if mod.get("links"):
#                 for (name, url) in mod["links"]:
#                     st.markdown(f"- [{name}]({url})")
#             st.markdown("**Mashqlar**")
#             checklist(f"{prefix}-mod-{i}", mod["exercises"], state_key)
#             st.markdown(f"**Deliverable**: `{mod['deliverable']}`")
#             st.text_area("Notes", key=f"notes_{prefix}_{i}")


# def kpi_progress(state_key: str, title: str) -> None:
#     state: Dict[str, bool] = st.session_state[state_key]
#     total = len(state)
#     done = sum(1 for v in state.values() if v)
#     pct = (done / total) * 100 if total else 0
#     st.metric(label=title, value=f"{done}/{total}", delta=f"{pct:.0f}%")


# # ----------------------
# # Pages
# # ----------------------

# def page_overview() -> None:
#     section_header("PlanA — 3 Yillik O‘quv Reja", "Asoslar → Data Science → Advanced ML/AI")
#     st.write(
#         "Bu platforma uch yillik rejani boshqarish, haftalik reja va 7 kunlik sprintlarni rejalashtirish, "
#         "hamda resurslar va portfolioni kuzatish uchun yaratilgan."
#     )
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         kpi_progress(PROGRESS_KEYS["y1"], "Yil 1 Progress")
#     with col2:
#         kpi_progress(PROGRESS_KEYS["y2"], "Yil 2 Progress")
#     with col3:
#         kpi_progress(PROGRESS_KEYS["y3"], "Yil 3 Progress")

#     st.divider()
#     st.markdown("### Asosiy manbalar (load-bearing)")
#     st.markdown("- MIT Linear Algebra (Strang)\n- Harvard CS50\n- Andrew Ng ML / deeplearning.ai\n- fast.ai Practical DL\n- ISLR (book + labs)")


# def page_year1() -> None:
#     section_header("Yil 1 — Asoslar", "Matematika, Python, CS, Linux/Git, SQL")
#     monthly_sections("Yil 1", YEAR1_MONTHLY, PROGRESS_KEYS["y1"])
#     module_sections("Yil 1 — Tafsilotli Modullar", MODULES_Y1, PROGRESS_KEYS["y1"], prefix="y1")


# def page_year2() -> None:
#     section_header("Yil 2 — Data Science & Applied Economics", "Statistika, ML, Econometrics, A/B, Deploy")
#     monthly_sections("Yil 2", YEAR2_MONTHLY, PROGRESS_KEYS["y2"])
#     module_sections("Yil 2 — Tafsilotli Modullar", MODULES_Y2, PROGRESS_KEYS["y2"], prefix="y2")


# def page_year3() -> None:
#     section_header("Yil 3 — Advanced ML / AI & Job Prep", "DL, NLP, MLOps, Research, Interviews")
#     monthly_sections("Yil 3", YEAR3_MONTHLY, PROGRESS_KEYS["y3"])
#     module_sections("Yil 3 — Tafsilotli Modullar", MODULES_Y3, PROGRESS_KEYS["y3"], prefix="y3")


# def page_sprint_7d() -> None:
#     section_header("7 kunlik Sprint", "Darhol boshlash uchun konkret reja")
#     data = st.session_state[PROGRESS_KEYS["sprint"]]
#     for day in DAYS_7:
#         with st.expander(day, expanded=False):
#             data[day]["task"] = st.text_input("Vazifa", value=data[day]["task"], key=f"sp_task_{day}")
#             data[day]["notes"] = st.text_area("Izohlar", value=data[day]["notes"], key=f"sp_notes_{day}")
#             data[day]["done"] = st.checkbox("Bajarildi", value=data[day]["done"], key=f"sp_done_{day}")
#     done = sum(1 for d in data.values() if d["done"])
#     st.success(f"Yakun: {done}/7 kun bajarildi")


# def page_weekly_plan() -> None:
#     section_header("Haftalik o‘qish jadvali", "Ish/ta’limga mos fleksibl shablon")
#     plan = st.session_state[PROGRESS_KEYS["weekly"]]
#     for day in WEEK_DAYS:
#         with st.expander(day, expanded=False):
#             blocks = plan[day]
#             for i, block in enumerate(blocks):
#                 cols = st.columns([1, 3])
#                 with cols[0]:
#                     st.text_input("Vaqt/Slot", value=block["slot"], key=f"wk_slot_{day}_{i}")
#                 with cols[1]:
#                     block["topic"] = st.text_input("Mavzu", value=block["topic"], key=f"wk_topic_{day}_{i}")

#     st.info("Maslahat: Google Calendar/Notion bilan sinxronlashtiring.")


# def page_portfolio_resources() -> None:
#     section_header("Portfolio & Resurslar")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("### Portfolio Checklist")
#         checklist("portfolio", PORTFOLIO_ITEMS, PROGRESS_KEYS["portfolio"])

#     with col2:
#         st.markdown("### Resource Hub")
#         res_state: Dict[str, bool] = st.session_state[PROGRESS_KEYS["resources"]]
#         for i, (name, url) in enumerate(RESOURCES):
#             cols = st.columns([4, 1])
#             with cols[0]:
#                 st.markdown(f"- [{name}]({url})")
#             with cols[1]:
#                 key = f"res:{name}"
#                 val = res_state.get(key, False)
#                 res_state[key] = st.checkbox("Seen", value=val, key=f"res_seen_{i}")

#     st.divider()
#     st.markdown("### Majburiy amaliy loyihalar (kamida 8 ta)")
#     must_projects = [
#         "Data cleaning & EDA — blog post + Kaggle dataset",
#         "Regression — price prediction + explainability",
#         "Classification — churn/health + metrics report",
#         "Time-series — ARIMA/Prophet/NN",
#         "NLP — sentiment/summarization (deploy)",
#         "CV — image classify + transfer learning",
#         "End-to-end deployed ML API — frontend demo",
#         "Open-source contribution yoki reproduced paper",
#     ]
#     checklist("projects", must_projects, PROGRESS_KEYS["portfolio"])


# # ----------------------
# # Main
# # ----------------------

# def main() -> None:
#     st.set_page_config(page_title="PlanA — 3-Year ML/AI", layout="wide")
#     ensure_session_state_defaults()

#     with st.sidebar:
#         st.markdown("### Navigatsiya")
#         page = st.radio(
#             "Bo‘limni tanlang",
#             (
#                 "Overview",
#                 "Yil 1",
#                 "Yil 2",
#                 "Yil 3",
#                 "7 kunlik Sprint",
#                 "Haftalik Reja",
#                 "Portfolio & Resurslar",
#             ),
#             index=0,
#         )
#         st.divider()
#         st.caption("Session-based progress. Persistent storage emas (hozircha).")

#     if page == "Overview":
#         page_overview()
#     elif page == "Yil 1":
#         page_year1()
#     elif page == "Yil 2":
#         page_year2()
#     elif page == "Yil 3":
#         page_year3()
#     elif page == "7 kunlik Sprint":
#         page_sprint_7d()
#     elif page == "Haftalik Reja":
#         page_weekly_plan()
#     elif page == "Portfolio & Resurslar":
#         page_portfolio_resources()


# if __name__ == "__main__":
#     main()







import io
import base64
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="0 → TOP 1% AI Engineer Roadmap", layout="wide")

st.title("0 → TOP 1% AI Engineer: 12-Month Plan, Portfolio, Tasks & Progress")
st.caption("Prepared for O'ktambek · Streamlit app")

# --------------------------
# Embedded Data (core roadmap/portfolio/resources)
# --------------------------
months_data = [
    {"Month": 1, "Focus": "Python Core, Git, CLI, Productivity", "Skills": "Python syntax, data structures, functions, OOP basics, Git/GitHub, virtualenv/conda, CLI", "Weekly Plan": "W1: Python syntax & data types • W2: Functions, modules, OOP intro • W3: Git/GitHub, CLI, tooling • W4: Mini project & code review", "Deliverables": "CLI Personal Assistant (argparse), Git workflow (PRs), README", "Checkpoint": "100 LeetCode/Eolymp easy tasks OR 30 HackerRank problems", "Est. Hours": 80},
    {"Month": 2, "Focus": "Math Foundations I (Linear Algebra & Calculus)", "Skills": "Vectors, matrices, eigenvalues, derivatives, gradient, chain rule", "Weekly Plan": "W1: Vectors, matrix ops • W2: Eigen/orthogonality • W3: Limits, derivatives • W4: Gradient descent from scratch", "Deliverables": "Numpy-based gradient descent & linear regression from scratch", "Checkpoint": "Khan Academy quizzes ≥80%", "Est. Hours": 80},
    {"Month": 3, "Focus": "Math Foundations II (Probability & Statistics) + EDA", "Skills": "Distributions, estimation, hypothesis testing, EDA with Pandas/Matplotlib", "Weekly Plan": "W1: Descriptive stats • W2: Probability & distributions • W3: Hypothesis testing • W4: EDA patterns & visual storytelling", "Deliverables": "EDA Notebook on 2 datasets + report", "Checkpoint": "Mini exam: A/B test simulation notebook", "Est. Hours": 80},
    {"Month": 4, "Focus": "Data Engineering Basics + SQL", "Skills": "SQL (JOIN/CTE/Window), ETL, data quality, Airflow intro", "Weekly Plan": "W1: SQL basics • W2: Advanced SQL (window, CTE) • W3: ETL with Python • W4: Airflow DAGs & data validation", "Deliverables": "ETL pipeline (CSV→Postgres) + 10 analytics queries", "Checkpoint": "HackerRank SQL (Gold) or ≥200 points", "Est. Hours": 80},
    {"Month": 5, "Focus": "Classical ML I", "Skills": "Regression, regularization, cross-val, metrics, feature engineering", "Weekly Plan": "W1: Linear/Ridge/Lasso • W2: Pipelines, CV, metrics • W3: Feature eng • W4: Project 1", "Deliverables": "Car Price Prediction pipeline (sklearn) + report", "Checkpoint": "RMSE/MAE vs baseline ≥30% improvement", "Est. Hours": 90},
    {"Month": 6, "Focus": "Classical ML II", "Skills": "Trees, RandomForest, XGBoost/LightGBM, model explainability (SHAP/LIME)", "Weekly Plan": "W1: Trees/ensembles • W2: Boosting • W3: Explainability • W4: Project 2", "Deliverables": "Medical Diagnosis classifier + SHAP analysis", "Checkpoint": "AUC ≥0.90 on holdout; proper calibration", "Est. Hours": 90},
    {"Month": 7, "Focus": "Deep Learning Fundamentals", "Skills": "PyTorch, backprop, optimization, regularization, training loops, callbacks", "Weekly Plan": "W1: NN & training loop • W2: Regularization, schedulers • W3: CNN basics • W4: Project 3", "Deliverables": "From-scratch MLP + PyTorch training framework", "Checkpoint": "Reproduce MNIST ≥99% acc", "Est. Hours": 90},
    {"Month": 8, "Focus": "Computer Vision Track", "Skills": "CNN architectures (ResNet, EfficientNet), transfer learning, YOLOv8", "Weekly Plan": "W1: Transfer learning • W2: ResNet/EfficientNet • W3: Object detection (YOLO) • W4: Project 4", "Deliverables": "Custom dataset classifier + YOLO detector", "Checkpoint": "Top-20% on a CV Kaggle/competition", "Est. Hours": 100},
    {"Month": 9, "Focus": "NLP & LLMs Track", "Skills": "Tokenization, Transformers, BERT, finetuning, evaluation, RAG", "Weekly Plan": "W1: Classical NLP • W2: Transformers/BERT • W3: Finetuning & eval • W4: RAG chatbot project", "Deliverables": "Domain RAG chatbot (LangChain + FAISS/Chroma)", "Checkpoint": "Response accuracy ≥85% on eval set", "Est. Hours": 100},
    {"Month": 10, "Focus": "MLOps & System Design", "Skills": "FastAPI, Docker, CI/CD, monitoring, batch/stream, data versioning (DVC)", "Weekly Plan": "W1: Serving (FastAPI) • W2: Docker & K8s basics • W3: CI/CD & tests • W4: Monitoring (Prometheus/Grafana)", "Deliverables": "Production-grade API for Month 6/9 model + CI/CD", "Checkpoint": "99.9% uptime on local tests; load test ≥200 RPS", "Est. Hours": 100},
    {"Month": 11, "Focus": "Time Series / Recommenders / RL (choose 1–2)", "Skills": "ARIMA/Prophet or matrix factorization/LightFM or DQN/Policy Gradients", "Weekly Plan": "W1: Theory • W2: Implementation • W3: Project • W4: Paper reproduction", "Deliverables": "One specialized project + paper reproduction", "Checkpoint": "SOTA reproduction within ±5%", "Est. Hours": 90},
    {"Month": 12, "Focus": "Capstone & Leadership", "Skills": "End-to-end system design, documentation, team practices, mentoring", "Weekly Plan": "W1: Capstone design • W2: Build • W3: Evaluate & optimize • W4: Writeup, demo, blog", "Deliverables": "Capstone SaaS app + technical report + demo video", "Checkpoint": "Code review by 2+ engineers; user testing feedback", "Est. Hours": 110},
]

portfolio_data = [
    {"Stage": 1, "Project": "EDA Storytelling Dashboard", "Domain": "Data Analysis", "Tech": "Pandas, Plotly/Streamlit", "Outcome": "Insight-rich dashboard + report"},
    {"Stage": 1, "Project": "SQL Analytics Pack", "Domain": "Analytics Engineering", "Tech": "Postgres, SQL, dbt (optional)", "Outcome": "10+ business queries + schema diagram"},
    {"Stage": 2, "Project": "Car Price Prediction", "Domain": "Supervised ML", "Tech": "sklearn, pipelines, joblib", "Outcome": "Deployed pipeline + README"},
    {"Stage": 2, "Project": "Medical Diagnosis Classifier", "Domain": "Healthcare ML", "Tech": "XGBoost/LightGBM, SHAP", "Outcome": "AUC≥0.90 + explainability"},
    {"Stage": 3, "Project": "MNIST from Scratch", "Domain": "DL Fundamentals", "Tech": "NumPy, PyTorch", "Outcome": "99%+ accuracy + clean training loop"},
    {"Stage": 3, "Project": "Custom CV Classifier", "Domain": "Computer Vision", "Tech": "PyTorch, Transfer Learning", "Outcome": ">90% acc on custom dataset"},
    {"Stage": 3, "Project": "Object Detection (YOLO)", "Domain": "Computer Vision", "Tech": "YOLOv8", "Outcome": "mAP benchmark + demo video"},
    {"Stage": 4, "Project": "Sentiment/BERT Finetune", "Domain": "NLP", "Tech": "Transformers/HF", "Outcome": "F1≥0.9 on domain dataset"},
    {"Stage": 4, "Project": "RAG Chatbot", "Domain": "LLM Apps", "Tech": "LangChain, FAISS/Chroma", "Outcome": "Retrieval eval + latency <500ms"},
    {"Stage": 5, "Project": "MLOps: Model Serving API", "Domain": "MLOps", "Tech": "FastAPI, Docker, CI/CD", "Outcome": "Containerized API + tests + monitoring"},
    {"Stage": 5, "Project": "Specialization (TS/Rec/RL)", "Domain": "Choose 1", "Tech": "Prophet/LightFM/DQN", "Outcome": "Paper reproduction ±5% of SOTA"},
    {"Stage": 5, "Project": "Capstone SaaS", "Domain": "Full-stack AI", "Tech": "Streamlit/Next.js + FastAPI + Cloud", "Outcome": "Live demo + paying/test users"}
]

resources_data = [
    {"Category": "Python & CS", "Name": "Python for Everybody", "Type": "Course", "URL": "https://www.coursera.org/specializations/python"},
    {"Category": "Python & CS", "Name": "Automate the Boring Stuff", "Type": "Book/Course", "URL": "https://automatetheboringstuff.com/"},
    {"Category": "Math", "Name": "Khan Academy (LA/Calc/Stats)", "Type": "Course", "URL": "https://www.khanacademy.org/"},
    {"Category": "Math", "Name": "Essence of Linear Algebra", "Type": "Video", "URL": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"},
    {"Category": "ML", "Name": "Hands-On ML (Géron)", "Type": "Book", "URL": "https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"},
    {"Category": "ML", "Name": "Kaggle Courses", "Type": "Course", "URL": "https://www.kaggle.com/learn"},
    {"Category": "DL", "Name": "fast.ai Practical DL", "Type": "Course", "URL": "https://course.fast.ai/"},
    {"Category": "DL", "Name": "Karpathy: Zero to Hero", "Type": "Video", "URL": "https://www.youtube.com/@AndrejKarpathy"},
    {"Category": "CV", "Name": "CS231n", "Type": "Course", "URL": "http://cs231n.stanford.edu/"},
    {"Category": "NLP/LLM", "Name": "CS224n", "Type": "Course", "URL": "http://web.stanford.edu/class/cs224n/"},
    {"Category": "NLP/LLM", "Name": "Hugging Face Course", "Type": "Course", "URL": "https://huggingface.co/learn"},
    {"Category": "MLOps", "Name": "Made With ML (MLOps)", "Type": "Course", "URL": "https://madewithml.com/"},
    {"Category": "MLOps", "Name": "Full-stack Deep Learning", "Type": "Course", "URL": "https://fullstackdeeplearning.com/"},
    {"Category": "System Design", "Name": "System Design Primer", "Type": "Repo", "URL": "https://github.com/donnemartin/system-design-primer"},
    {"Category": "Deployment", "Name": "FastAPI", "Type": "Docs", "URL": "https://fastapi.tiangolo.com/"},
    {"Category": "Deployment", "Name": "Docker", "Type": "Docs", "URL": "https://docs.docker.com/"},
    {"Category": "Tools", "Name": "LangChain Docs", "Type": "Docs", "URL": "https://python.langchain.com/"},
    {"Category": "Tools", "Name": "FAISS", "Type": "Repo", "URL": "https://github.com/facebookresearch/faiss"},
    {"Category": "Research", "Name": "arXiv Sanity (by Karpathy)", "Type": "Tool", "URL": "http://www.arxiv-sanity.com/"},
    {"Category": "Communities", "Name": "Kaggle", "Type": "Community", "URL": "https://www.kaggle.com/"},
]

plan_df = pd.DataFrame(months_data)
portfolio_df = pd.DataFrame(portfolio_data)
resources_df = pd.DataFrame(resources_data)

def df_download_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=file_name, mime="text/csv")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📅 12-Month Plan", "🧰 Portfolio Roadmap", "📚 Resources", "✅ Checklists",
    "🗓️ Weekly Tasks", "📈 Progress", "🔥 Burndown", "🎯 OKR & 🧩 Habits"
])

# --------------------------
# Tab 1: Study Plan
# --------------------------
with tab1:
    st.subheader("12-Month Study Plan")
    with st.expander("Filter by Focus or Min Hours", expanded=True):
        focus_query = st.text_input("Search in Focus/Skills", "")
        min_hours = st.slider("Minimum Estimated Hours", 0, 120, 0, step=10)
    filtered = plan_df.copy()
    if focus_query:
        mask = filtered["Focus"].str.contains(focus_query, case=False) | filtered["Skills"].str.contains(focus_query, case=False)
        filtered = filtered[mask]
    filtered = filtered[filtered["Est. Hours"] >= min_hours]
    st.dataframe(filtered, use_container_width=True)
    df_download_button(filtered, "Download Plan (CSV)", "plan_12_months.csv")

    st.markdown("### Milestone Suggestions")
    st.markdown("""
    - **Monthly**: 1 shipped project + 1 technical write-up
    - **Quarterly**: Public demo (Streamlit/Space) + code review by peers
    - **Metrics**: Model quality vs. baselines, latency targets, uptime, test coverage
    """)

# --------------------------
# Tab 2: Portfolio Roadmap
# --------------------------
with tab2:
    st.subheader("Portfolio Roadmap")
    col1, col2 = st.columns([1,1])
    with col1:
        stage = st.selectbox("Stage filter", ["All"] + sorted(set(portfolio_df["Stage"])))
    with col2:
        domain = st.selectbox("Domain filter", ["All"] + sorted(set(portfolio_df["Domain"])))
    filtered_pf = portfolio_df.copy()
    if stage != "All":
        filtered_pf = filtered_pf[filtered_pf["Stage"] == stage]
    if domain != "All":
        filtered_pf = filtered_pf[filtered_pf["Domain"] == domain]
    st.dataframe(filtered_pf, use_container_width=True)
    df_download_button(filtered_pf, "Download Portfolio (CSV)", "portfolio_roadmap.csv")

    st.markdown("### Repo Structure Template")
    st.code("""
    repo/
      ├─ src/
      │   ├─ data/
      │   ├─ features/
      │   ├─ models/
      │   └─ serving/
      ├─ notebooks/
      ├─ tests/
      ├─ docker/
      ├─ configs/
      ├─ Makefile
      ├─ requirements.txt
      └─ README.md
    """, language="text")

# --------------------------
# Tab 3: Resources
# --------------------------
with tab3:
    st.subheader("Curated Resources")
    category = st.selectbox("Filter by Category", ["All"] + sorted(set(resources_df["Category"])))
    rdf = resources_df.copy()
    if category != "All":
        rdf = rdf[rdf["Category"] == category]
    st.dataframe(rdf, use_container_width=True)
    df_download_button(rdf, "Download Resources (CSV)", "resources_list.csv")

    st.markdown("### Study Cadence")
    st.markdown("""
    - Daily: 1h theory · 2h coding · 1h reading
    - Weekly: 1 project increment, 1 code review, 1 blog post draft
    - Monthly: Public demo + retrospective
    """)

# --------------------------
# Tab 4: Checklists
# --------------------------
with tab4:
    st.subheader("Execution Checklists")
    st.markdown("""
    **Every Project**
    - [ ] Clear problem statement & baseline
    - [ ] Reproducible data pipeline
    - [ ] Evaluation with proper metrics
    - [ ] Unit tests + CI
    - [ ] Dockerized service / Streamlit demo
    - [ ] README with results & lessons
    """)
    st.markdown("""
    **MLOps Readiness**
    - [ ] FastAPI endpoint with input validation
    - [ ] Logging & monitoring (latency, errors)
    - [ ] Model registry & versioning
    - [ ] Rollback plan
    """)

# --------------------------
# Tab 5: Weekly Tasks (Tracker)
# --------------------------
with tab5:
    st.subheader("Weekly Task Tracker")
    st.write("Upload your existing CSV or start from the template and edit inline.")

    default_week = (dt.date.today() - dt.timedelta(days=dt.date.today().weekday())).isoformat()
    template = pd.DataFrame([
        {"WeekStart": default_week, "Task": "Finish Kaggle EDA", "Category": "Analysis", "EstHours": 6, "Status": "To Do", "Notes": ""},
        {"WeekStart": default_week, "Task": "Implement validation split", "Category": "ML", "EstHours": 4, "Status": "In Progress", "Notes": ""},
        {"WeekStart": default_week, "Task": "Write blog: Regularization", "Category": "Writing", "EstHours": 3, "Status": "To Do", "Notes": ""},
    ])

    uploaded = st.file_uploader("Upload Weekly Tasks CSV", type=["csv"])
    if "tasks_df" not in st.session_state:
        st.session_state["tasks_df"] = template

    if uploaded is not None:
        try:
            st.session_state["tasks_df"] = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    edited = st.data_editor(st.session_state["tasks_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["tasks_df"] = edited

    csv = edited.to_csv(index=False).encode("utf-8")
    st.download_button("Download Weekly Tasks (CSV)", data=csv, file_name="weekly_tasks.csv", mime="text/csv")

    if not edited.empty:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Tasks (Total)", len(edited))
        with col_b:
            done = (edited["Status"].astype(str).str.lower() == "done").sum()
            st.metric("Done", int(done))
        with col_c:
            st.metric("Est. Hours", int(edited["EstHours"].fillna(0).sum()))

# --------------------------
# Tab 6: Progress (Analytics)
# --------------------------
with tab6:
    st.subheader("Progress Analytics")
    st.write("Track your study hours and shipped projects per week.")

    progress_template = pd.DataFrame([
        {"WeekStart": default_week, "StudyHours": 20, "ProjectsShipped": 1, "Commits": 15, "LessonsRead": 5},
        {"WeekStart": (dt.date.fromisoformat(default_week) - dt.timedelta(days=7)).isoformat(), "StudyHours": 18, "ProjectsShipped": 0, "Commits": 12, "LessonsRead": 4},
    ])

    prog_upload = st.file_uploader("Upload Progress CSV", type=["csv"], key="prog_upload")
    if "progress_df" not in st.session_state:
        st.session_state["progress_df"] = progress_template

    if prog_upload is not None:
        try:
            st.session_state["progress_df"] = pd.read_csv(prog_upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    prog_df = st.data_editor(st.session_state["progress_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["progress_df"] = prog_df

    if not prog_df.empty:
        try:
            prog_df["WeekStart"] = pd.to_datetime(prog_df["WeekStart"]).dt.date
            prog_df = prog_df.sort_values("WeekStart")
        except Exception as e:
            st.warning(f"Date parsing issue: {e}")

    target_hours = st.number_input("Weekly Target Study Hours", min_value=1, max_value=100, value=25, step=1)
    target_projects = st.number_input("Quarterly Project Target", min_value=1, max_value=12, value=3, step=1)

    if not prog_df.empty and "StudyHours" in prog_df.columns:
        fig1 = plt.figure()
        plt.plot(prog_df["WeekStart"], prog_df["StudyHours"], marker="o", label="StudyHours")
        plt.axhline(y=target_hours, linestyle='--', color='red', label='Target')
        plt.title("Weekly Study Hours vs Target")
        plt.xlabel("Week Start")
        plt.ylabel("Hours")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

    if not prog_df.empty and "ProjectsShipped" in prog_df.columns:
        tmp = prog_df.copy()
        tmp["CumProjects"] = tmp["ProjectsShipped"].fillna(0).cumsum()
        fig2 = plt.figure()
        plt.plot(tmp["WeekStart"], tmp["CumProjects"], marker="o")
        plt.title("Cumulative Projects Shipped")
        plt.xlabel("Week Start")
        plt.ylabel("Cumulative Projects")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

    csv_prog = prog_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Progress (CSV)", data=csv_prog, file_name="progress_tracker.csv", mime="text/csv")

    st.markdown("**Tip:** Use Monday as WeekStart to keep trends accurate.")

# --------------------------
# Tab 7: Burndown (Weekly)
# --------------------------
with tab7:
    st.subheader("Weekly Burndown Chart")
    st.write("Plan your week, log daily hours, and compare actual vs ideal burndown.")

    # Select week
    week_dates = sorted(set(pd.to_datetime(st.session_state.get("tasks_df", pd.DataFrame({"WeekStart": [default_week]}))["WeekStart"]).dt.date)) if "tasks_df" in st.session_state else [dt.date.fromisoformat(default_week)]
    chosen_week = st.selectbox("WeekStart (Monday)", week_dates, index=len(week_dates)-1)

    # Filter tasks for chosen week
    tasks = st.session_state.get("tasks_df", pd.DataFrame())
    if not tasks.empty:
        try:
            tasks["WeekStart"] = pd.to_datetime(tasks["WeekStart"]).dt.date
        except Exception:
            pass
    week_tasks = tasks[tasks["WeekStart"] == chosen_week] if not tasks.empty else pd.DataFrame(columns=["Task","EstHours","Status"])

    total_est = float(week_tasks.get("EstHours", pd.Series(dtype=float)).fillna(0).sum())
    st.metric("Estimated Hours (week)", int(total_est))

    # Daily log editor
    monday = chosen_week
    days = [monday + dt.timedelta(days=i) for i in range(7)]
    default_log = pd.DataFrame([{"Date": d.isoformat(), "HoursDone": 0.0} for d in days])

    burn_upload = st.file_uploader("Upload Daily Hours CSV (Date, HoursDone)", type=["csv"], key="burn_upload")
    if "burn_log" not in st.session_state:
        st.session_state["burn_log"] = default_log

    if burn_upload is not None:
        try:
            st.session_state["burn_log"] = pd.read_csv(burn_upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    log_df = st.data_editor(st.session_state["burn_log"], num_rows="dynamic", use_container_width=True)
    st.session_state["burn_log"] = log_df

    # Compute remaining hours per day
    try:
        log_df["Date"] = pd.to_datetime(log_df["Date"]).dt.date
        log_df = log_df.sort_values("Date")
    except Exception as e:
        st.warning(f"Date parsing issue: {e}")

    cum_done = log_df["HoursDone"].fillna(0).cumsum() if not log_df.empty else pd.Series([0]*7)
    remaining = [max(total_est - x, 0) for x in cum_done]

    # Ideal line
    ideal = []
    if total_est > 0 and len(days) > 0:
        for i in range(len(days)):
            ideal.append(total_est * (1 - i/(len(days)-1))) if len(days) > 1 else ideal.append(0)

    # Plot burndown
    if len(remaining) > 0:
        fig3 = plt.figure()
        plt.plot(days[:len(remaining)], remaining, marker="o", label="Remaining")
        if ideal:
            plt.plot(days[:len(ideal)], ideal, linestyle='--', label="Ideal")
        plt.title("Burndown: Remaining Hours (Actual vs Ideal)")
        plt.xlabel("Date")
        plt.ylabel("Remaining Hours")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

    # Download log
    csv_burn = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Burndown Log (CSV)", data=csv_burn, file_name="burndown_log.csv", mime="text/csv")

# --------------------------
# Tab 8: OKR & Habits
# --------------------------
with tab8:
    st.subheader("Quarterly OKRs")
    okr_template = pd.DataFrame([
        {"Quarter": "2025-Q4", "Objective": "Ship Capstone MVP", "KR": "MVP live users", "Target": 50, "Current": 10, "Confidence": 0.6, "Owner": "O'ktambek"},
        {"Quarter": "2025-Q4", "Objective": "Strengthen ML Fundamentals", "KR": "Practice problems solved", "Target": 200, "Current": 80, "Confidence": 0.7, "Owner": "O'ktambek"},
    ])

    okr_upload = st.file_uploader("Upload OKRs CSV", type=["csv"], key="okr_upload")
    if "okr_df" not in st.session_state:
        st.session_state["okr_df"] = okr_template

    if okr_upload is not None:
        try:
            st.session_state["okr_df"] = pd.read_csv(okr_upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    okr_df = st.data_editor(st.session_state["okr_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["okr_df"] = okr_df

    # Select a KR to visualize
    if not okr_df.empty:
        sel_idx = st.number_input("Select KR row to visualize (0-based)", min_value=0, max_value=len(okr_df)-1, value=0, step=1)
        row = okr_df.iloc[int(sel_idx)]
        target = float(row.get("Target", 0) or 0)
        current = float(row.get("Current", 0) or 0)

        fig4 = plt.figure()
        plt.bar(["Current", "Target"], [current, target], color=['#1f77b4', '#ff7f0e'])
        plt.title(f"KR Progress: {row.get('KR', '')}")
        plt.ylabel("Value")
        plt.tight_layout()
        st.pyplot(fig4)

    csv_okr = okr_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download OKRs (CSV)", data=csv_okr, file_name="okrs.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Daily Habit Tracker")
    today = dt.date.today()
    start_default = today - dt.timedelta(days=6)
    habits_template = pd.DataFrame([
        {"Date": (start_default + dt.timedelta(days=i)).isoformat(), "TheoryMin": 60, "CodingMin": 120, "ReadingMin": 60, "ExerciseMin": 20}
        for i in range(7)
    ])

    h_upload = st.file_uploader("Upload Habits CSV", type=["csv"], key="habits_upload")
    if "habits_df" not in st.session_state:
        st.session_state["habits_df"] = habits_template

    if h_upload is not None:
        try:
            st.session_state["habits_df"] = pd.read_csv(h_upload)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    habits_df = st.data_editor(st.session_state["habits_df"], num_rows="dynamic", use_container_width=True)
    st.session_state["habits_df"] = habits_df

    # Goals
    st.markdown("**Daily Goals (minutes)**")
    c1, c2, c3, c4 = st.columns(4)
    g_theory = c1.number_input("Theory", min_value=0, max_value=600, value=60, step=10)
    g_coding = c2.number_input("Coding", min_value=0, max_value=600, value=120, step=10)
    g_read = c3.number_input("Reading", min_value=0, max_value=600, value=60, step=10)
    g_ex = c4.number_input("Exercise", min_value=0, max_value=300, value=20, step=5)

    # Compute daily percent to goal
    try:
        habits_df["Date"] = pd.to_datetime(habits_df["Date"]).dt.date
        habits_df = habits_df.sort_values("Date")
    except Exception as e:
        st.warning(f"Date parsing issue: {e}")

    def pct(a, b):
        return (a / b * 100.0) if b and b > 0 else 0.0

    pct_goal = []
    for _, r in habits_df.iterrows():
        p = (
            pct(r.get("TheoryMin", 0), g_theory) +
            pct(r.get("CodingMin", 0), g_coding) +
            pct(r.get("ReadingMin", 0), g_read) +
            pct(r.get("ExerciseMin", 0), g_ex)
        ) / 4.0
        pct_goal.append(p)

    # Plot habit adherence
    if len(pct_goal) > 0:
        fig5 = plt.figure()
        plt.plot(habits_df["Date"], pct_goal, marker="o", color='green')
        plt.axhline(y=100, linestyle='--', color='red', label='100% Goal')
        plt.title("Daily Habit Adherence (% of Goal)")
        plt.xlabel("Date")
        plt.ylabel("% Goal")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig5)

    csv_h = habits_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Habits (CSV)", data=csv_h, file_name="habits.csv", mime="text/csv")
