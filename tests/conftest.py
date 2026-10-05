import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ["unada-labs/enterprise-rag-chatbot", "unada-labs/databricks-image-ingestion",
               "unada-labs/cortex-vision-platform", "mars/venus-ml-models",
               "mars/document-summarizer-chatbot"]:
    sys.path.insert(0, str(ROOT / folder))
