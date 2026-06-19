# Overview

This document outlines the engineering and execution plan for building the automated multi-modal evidence review agent. To balance speed, performance, and budget constraints, this solution leverages a hybrid multi-modal strategy: development and testing are conducted entirely using Gemini 3.1 Flash-Lite (via Google AI Studio free tier), while production execution utilizes DeepSeek V4 Pro (via the DeepSeek API) for ultra-low latency, full concurrency, and massive throughput.

The implementation strictly uses the Pydantic AI framework to enforce type safety, dynamic tool-driven context injection, and rigid structured JSON schema mapping.  

## Phase 1: Data Ingestion & Schema Definition

To satisfy the strict formatting and structural requirements of problem_statement.md, we establish programmatic safeguards before hitting any API endpoints.

### 1.1 Local Context Preloading

Load contextual datasets (dataset/user_history.csv and dataset/evidence_requirements.csv) into Pandas DataFrames or memory-mapped dictionaries.  Implement rapid indexing lookups mapped to user_id and claim_object respectively to bypass injecting massive global csv files into the LLM context window.

### 1.2 Output Type Constraints (Pydantic Model)
Define a core output schema using Pydantic fields to handle exact constraints:  Literal Categorization: Map fields exactly to allowed strings (e.g., claim_status restricted to ['supported', 'contradicted', 'not_enough_information']).  Object Part Alignment: Implement custom validation or separate conditional rules ensuring the object_part field maps strictly to the specified object type (e.g., a laptop claim must reject vehicle components like front_bumper).

### 1.3 Multimodal File Utility
Build an image-loading utility that handles semicolon-separated paths in image_paths.  Convert localized media files into raw byte formats supported by both the Google AI Studio and OpenAI-compatible DeepSeek vision payloads.

## Phase 2: Modular Architecture Design (Pydantic AI)

The system architecture follows an generally successful pattern: a single robust agent backed by structured tool loops, deterministic validation, and strict schema-enforced JSON out-routing rather than complex, highly error-prone multi-agent workflows.

[ Input Row from claims.csv ]
↓
[ Pydantic AI Client Factory ] ← Switch Model (Gemini / DeepSeek)
↓
[ Single Routing Agent ] ← Local Tool: User History Data & Evidence Checklists
↓
[ Pydantic JSON Validation ] → [Fails] → Auto-Retry Loop
↓
[Passes][ Final output.csv Row ]

### 2.1 Pydantic AI Client Factory
Configure a client wrapper capable of seamless swapping via environment variables or a runtime toggle switch:  Dev/Test Provider: gemini-3.1-flash-lite via pydantic_ai.models.gemini.  Prod Provider: OpenAI-compatible base URL pointing to the deepseek-v4-pro vision endpoints.  

### 2.2 Local Context Extraction Tools
Utilize Pydantic AI's @agent.tool decorators to enable local context injection dynamically per row execution:  `get_user_risk_context(user_id: str)`: Pulls historical metrics and summary risk tags to inject as structural metadata.  `get_minimum_evidence(claim_object: str)`: Extracts the subset checklist requirements before triggering model inference.  

## Phase 3: Prompt Crafting & Optimization Strategy
The core reasoning capability must be robust across models. To maximize performance, we organize the system prompt based on specific alignment heuristics.

### 3.1 Hierarchy of Truth Anchor
Explicitly instruct the model that visual files are the absolute primary source of truth.  Define boundaries for contradictions: if text implies an issue but the visual metadata shows zero damage, the status must resolve to contradicted.  

### 3.2 Prompt Caching Layout
For production execution with DeepSeek V4, group all static operational criteria, structural definitions, and allowable enum bounds at the very top of the system text block. This triggers DeepSeek's automatic prompt caching mechanism, drastically cutting text processing fees by up to 90%.

## Phase 4: Execution & Rate-Limit Management

### 4.1 Dev / Iterative Loop (Gemini 3.1 Flash-Lite)
Execute processing over dataset/sample_claims.csv.  Combine Pydantic AI's internal execution loops with tenacity exponential backoff wrappers to cleanly catch and mitigate 429 Too Many Requests or quota exceptions.  Introduce an explicit asynchronous throttling cadence (asyncio.sleep(4.0)) to ensure pipeline pacing stays comfortably within the free tier limit of 15 requests per minute.

### 4.2 Production Batch Loop (DeepSeek V4 Pro)
Switch client factory routing directly over to DeepSeek.  Shift from throttled serial pacing to a high-concurrency batch processor using asyncio.gather.  Tap into DeepSeek's massive concurrent processing thresholds to run all independent dataset/claims.csv evaluation routines across concurrent worker pools, cutting hours of execution down to single-digit minutes.  

## Phase 5: Telemetry, Verification, and Submission
The pipeline concludes with an automated collection loop.

### 5.1 Telemetry Collection Hook
Intercept every successful agent lifecycle payload to log usage metrics (input token volume, output token count, inference latency, image asset counts).  Track downstream processing discrepancies to feed statistics into the final reporting suite. 

### 5.2 Compilation and Artifact Packaging
- Compilation Step: Convert the aggregate collection of Pydantic object elements straight into an organized multi-column data sheet structured exactly like the defined 14-column layout in problem_statement.md. Save the structured file natively to output.csv.  
- Analysis Builder: Execute a post-processing script that translates accumulated operational telemetry straight into standard pricing math and performance tracking structures inside evaluation/evaluation_report.md.  
- Artifact Assembly: Zip up full executable scripts, operational environments, schemas, and verification logs into the terminal submission file (code.zip).  