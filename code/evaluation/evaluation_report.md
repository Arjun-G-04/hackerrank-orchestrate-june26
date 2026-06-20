# Multi-Modal Evidence Review Evaluation Report
 
**Evaluation Timestamp:** 2026-06-20 22:13:05
**Active Provider:** GCP
**Model Name:** gemini-2.5-flash

---

## 1. Accuracy Metrics

Evaluated on `20` labeled sample claims from `sample_claims.csv`.

| Field / Classification | Correct Predictions | Total Claims | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Evidence Standard Met** | 17 | 20 | 85.0% |
| **Valid Image** | 17 | 20 | 85.0% |
| **Claim Status** | 15 | 20 | 75.0% |
| **Issue Type** | 14 | 20 | 70.0% |
| **Object Part** | 14 | 20 | 70.0% |
| **Severity** | 13 | 20 | 65.0% |
| **Risk Flags** | 12 | 20 | 60.0% |

---

## 2. Operational Telemetry & Cost Analysis

### Performance Metrics
* **Total Evaluation Runtime:** 386.70 seconds
* **Average Latency per Claim:** 19.33 seconds
* **Total Model Calls:** 20 / 20 successful
* **Total Processed Images:** 29

### Token Volumes
* **Total Input Tokens:** 205,761
* **Total Output Tokens:** 38,124
* **Average Input Tokens per Request:** 10,288
* **Average Output Tokens per Request:** 1,906

### Cost Estimations
* **Estimated Cost (Sample Set):** $0.15704
* **Extrapolated Cost (100 Claims):** $0.7852 if total_rows else 0.0

*Notes on Pricing Model:*
GCP Vertex AI (Gemini 2.5 Flash Paid Tier) Pricing Assumptions:
- Input: $0.30 / 1M tokens
- Output (including thinking): $2.50 / 1M tokens
