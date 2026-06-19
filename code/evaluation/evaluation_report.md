# Multi-Modal Evidence Review Evaluation Report
 
**Evaluation Timestamp:** 2026-06-19 21:37:40
**Active Provider:** GCP
**Model Name:** gemini-2.5-flash

---

## 1. Accuracy Metrics

Evaluated on `20` labeled sample claims from `sample_claims.csv`.

| Field / Classification | Correct Predictions | Total Claims | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Evidence Standard Met** | 20 | 20 | 100.0% |
| **Valid Image** | 20 | 20 | 100.0% |
| **Claim Status** | 20 | 20 | 100.0% |
| **Issue Type** | 20 | 20 | 100.0% |
| **Object Part** | 20 | 20 | 100.0% |
| **Severity** | 20 | 20 | 100.0% |
| **Risk Flags** | 20 | 20 | 100.0% |

---

## 2. Operational Telemetry & Cost Analysis

### Performance Metrics
* **Total Evaluation Runtime:** 409.33 seconds
* **Average Latency per Claim:** 20.47 seconds
* **Total Model Calls:** 20 / 20 successful
* **Total Processed Images:** 29

### Token Volumes
* **Total Input Tokens:** 189,225
* **Total Output Tokens:** 40,821
* **Average Input Tokens per Request:** 9,461
* **Average Output Tokens per Request:** 2,041

### Cost Estimations
* **Estimated Cost (Sample Set):** $0.15882
* **Extrapolated Cost (100 Claims):** $0.7941 if total_rows else 0.0

*Notes on Pricing Model:*
GCP Vertex AI (Gemini 2.5 Flash Paid Tier) Pricing Assumptions:
- Input: $0.30 / 1M tokens
- Output (including thinking): $2.50 / 1M tokens