# My Internship Work: UNADA Labs and MaRS

This repo is a write-up of what I worked on during my two internships, along with small working versions of the core ideas so you can actually run them.

A quick note before you dig in: most of what I built was for clients or for internal company products, so I can't put the original code, data or model weights here. What you'll find instead is my own simplified rebuild of the main logic behind each project. The pieces that only make sense with the real systems (the 48 trained vision models, the client dashboards, the internal tools) are explained in the READMEs instead.

## Where I worked

| Company | Role | Dates |
|---|---|---|
| UNADA Labs Pvt. Ltd. | AI/ML Intern | Feb 2026 - Aug 2026 |
| MaRS Planning & Engineering Services | AI Intern | Mar 2025 - Apr 2025 |

I started at MaRS doing classical ML, mostly tree models on tabular data. At UNADA Labs the work got a lot broader: computer vision, RAG chatbots, LLM agents and a data pipeline on Databricks. Looking back, MaRS is where I learned to clean messy data and compare models properly, and UNADA is where I learned how these things behave in production.

## What's inside

### UNADA Labs
| Project | What it was | Code here? |
|---|---|---|
| [Cortex vision platform](unada-labs/cortex-vision-platform) | One platform running about 48 vision models for object detection in images and videos | Routing and post-processing logic |
| [RAG chatbots for a large enterprise client](unada-labs/enterprise-rag-chatbot) | Document Q&A chatbots with hybrid BM25 + vector search | Full retrieval pipeline |
| [LangChain agents](unada-labs/langchain-agents) | LLM agents that call tools to finish multi-step tasks | Write-up |
| [Enterprise dashboards](unada-labs/enterprise-dashboards) | Business dashboards for project and sales KPIs | Write-up |
| [Databricks image ingestion](unada-labs/databricks-image-ingestion) | Automated camera image pipeline with backfill and daily jobs | Full pipeline logic |
| [Generative AI tools](unada-labs/genai-visualization) | Render generation, moodboards and room staging | Write-up |

### MaRS
| Project | What it was | Code here? |
|---|---|---|
| [ML models for Venus](mars/venus-ml-models) | XGBoost, Random Forest and Decision Tree models for an internal platform | Full workflow on synthetic data |
| [Document summarizer chatbot](mars/document-summarizer-chatbot) | Summarize a document, then ask it questions | Working version |

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # 14 tests
```

Each project folder has its own script you can run directly, for example:

```bash
python unada-labs/enterprise-rag-chatbot/hybrid_retrieval.py
python unada-labs/databricks-image-ingestion/ingestion.py
python mars/venus-ml-models/train_compare.py
```

Nothing needs an API key or internet access once the packages are installed.

## Contact
Kanishq Tanwar, kanishqtanwar35@gmail.com, [LinkedIn](https://linkedin.com/in/kanishq-tanwar)
