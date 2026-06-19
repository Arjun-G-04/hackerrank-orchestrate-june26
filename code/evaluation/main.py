import os
import sys
import asyncio
import time
from pathlib import Path
import pandas as pd
from loguru import logger

# Add parent directory and current directory to sys.path to enable local imports
eval_dir = Path(__file__).resolve().parent
code_dir = eval_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
if str(eval_dir) not in sys.path:
    sys.path.append(str(eval_dir))

# Local imports
from client import get_model  # noqa: E402
from utils import load_context_data, REPO_ROOT  # noqa: E402
from main import process_row, agent  # noqa: E402


def compare_risk_flags(pred: str, truth: str) -> bool:
    """
    Compares risk flags as sets, ignoring ordering and whitespace.
    """
    pred_set = {f.strip().lower() for f in str(pred).split(";") if f.strip()}
    truth_set = {f.strip().lower() for f in str(truth).split(";") if f.strip()}
    return pred_set == truth_set


def compare_bools(pred: str, truth: str) -> bool:
    """
    Compares boolean values represented as strings.
    """
    p = str(pred).strip().lower()
    t = str(truth).strip().lower()
    return p == t


async def run_evaluation():
    # Paths configuration
    sample_csv_path = REPO_ROOT / "dataset" / "sample_claims.csv"
    report_output_path = eval_dir / "evaluation_report.md"

    logger.info(f"Starting evaluation on sample dataset: {sample_csv_path}")
    if not sample_csv_path.exists():
        logger.error(f"Sample claims file not found: {sample_csv_path}")
        sys.exit(1)

    # Pre-load context datasets
    load_context_data()

    # Initialize model client
    model = get_model()

    # Default parameters based on provider selection
    provider = os.getenv("ACTIVE_PROVIDER", "gcp").lower().strip()
    if provider in ("gemini", "gcp"):
        # Pace Google/Vertex providers safely to avoid TPM/RPM exhaustion
        concurrency = 1
        throttle = 4.0 if provider == "gemini" else 2.0
    elif provider == "nvidia":
        concurrency = 1
        throttle = 1.8
    else:
        concurrency = 5
        throttle = 0.0

    df_sample = pd.read_csv(sample_csv_path)
    logger.info(f"Loaded {len(df_sample)} sample rows for evaluation.")

    sem = asyncio.Semaphore(concurrency)
    tasks = []

    for _, row in df_sample.iterrows():
        tasks.append(process_row(row, agent, model, sem, throttle))

    start_time = time.time()
    predictions = await asyncio.gather(*tasks)
    total_eval_time = time.time() - start_time

    # Calculate metrics
    total_rows = len(df_sample)
    correct_evidence_standard_met = 0
    correct_valid_image = 0
    correct_claim_status = 0
    correct_issue_type = 0
    correct_object_part = 0
    correct_severity = 0
    correct_risk_flags = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_images = 0
    successful_calls = 0

    for idx, pred in enumerate(predictions):
        truth = df_sample.iloc[idx]

        # Token & image metrics
        total_input_tokens += pred["_input_tokens"]
        total_output_tokens += pred["_output_tokens"]
        total_images += pred["_images_count"]
        if pred["_success"]:
            successful_calls += 1

        # Accuracy comparison
        if compare_bools(pred["evidence_standard_met"], truth["evidence_standard_met"]):
            correct_evidence_standard_met += 1
        if compare_bools(pred["valid_image"], truth["valid_image"]):
            correct_valid_image += 1
        if (
            str(pred["claim_status"]).strip().lower()
            == str(truth["claim_status"]).strip().lower()
        ):
            correct_claim_status += 1
        if (
            str(pred["issue_type"]).strip().lower()
            == str(truth["issue_type"]).strip().lower()
        ):
            correct_issue_type += 1
        if (
            str(pred["object_part"]).strip().lower()
            == str(truth["object_part"]).strip().lower()
        ):
            correct_object_part += 1
        if (
            str(pred["severity"]).strip().lower()
            == str(truth["severity"]).strip().lower()
        ):
            correct_severity += 1
        if compare_risk_flags(pred["risk_flags"], truth["risk_flags"]):
            correct_risk_flags += 1

    # Pricing assumptions
    # Gemini 2.5 Flash Paid Tier: $0.30 per 1M input, $2.50 per 1M output (including thinking)
    cost_input = (total_input_tokens / 1_000_000) * 0.30
    cost_output = (total_output_tokens / 1_000_000) * 2.50
    total_cost = cost_input + cost_output
    pricing_notes = "GCP Vertex AI (Gemini 2.5 Flash Paid Tier) Pricing Assumptions:\n- Input: $0.30 / 1M tokens\n- Output (including thinking): $2.50 / 1M tokens"

    # Determine model name for display
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    # Compile report
    report_content = f"""# Multi-Modal Evidence Review Evaluation Report
 
**Evaluation Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
**Active Provider:** {provider.upper()}
**Model Name:** {model_name}

---

## 1. Accuracy Metrics

Evaluated on `{total_rows}` labeled sample claims from `sample_claims.csv`.

| Field / Classification | Correct Predictions | Total Claims | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Evidence Standard Met** | {correct_evidence_standard_met} | {total_rows} | {(correct_evidence_standard_met / total_rows) * 100:.1f}% |
| **Valid Image** | {correct_valid_image} | {total_rows} | {(correct_valid_image / total_rows) * 100:.1f}% |
| **Claim Status** | {correct_claim_status} | {total_rows} | {(correct_claim_status / total_rows) * 100:.1f}% |
| **Issue Type** | {correct_issue_type} | {total_rows} | {(correct_issue_type / total_rows) * 100:.1f}% |
| **Object Part** | {correct_object_part} | {total_rows} | {(correct_object_part / total_rows) * 100:.1f}% |
| **Severity** | {correct_severity} | {total_rows} | {(correct_severity / total_rows) * 100:.1f}% |
| **Risk Flags** | {correct_risk_flags} | {total_rows} | {(correct_risk_flags / total_rows) * 100:.1f}% |

---

## 2. Operational Telemetry & Cost Analysis

### Performance Metrics
* **Total Evaluation Runtime:** {total_eval_time:.2f} seconds
* **Average Latency per Claim:** {total_eval_time / total_rows:.2f} seconds
* **Total Model Calls:** {successful_calls} / {total_rows} successful
* **Total Processed Images:** {total_images}

### Token Volumes
* **Total Input Tokens:** {total_input_tokens:,}
* **Total Output Tokens:** {total_output_tokens:,}
* **Average Input Tokens per Request:** {total_input_tokens // total_rows if total_rows else 0:,}
* **Average Output Tokens per Request:** {total_output_tokens // total_rows if total_rows else 0:,}

### Cost Estimations
* **Estimated Cost (Sample Set):** ${total_cost:.5f}
* **Extrapolated Cost (100 Claims):** ${(total_cost / total_rows) * 100:.4f} if total_rows else 0.0

*Notes on Pricing Model:*
{pricing_notes}
"""

    # Save to file
    with open(report_output_path, "w") as f:
        f.write(report_content)

    logger.success(f"Evaluation report written successfully to: {report_output_path}")


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
