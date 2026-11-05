# Instamart Customer Intelligence & Support Copilot

## 1. Overview

The **Instamart Customer Intelligence & Support Copilot** is an end-to-end Applied AI and Machine Learning project designed to improve customer retention, automate support-ticket classification, and assist customer-support teams with policy-based decisions.

The project combines data engineering, exploratory data analysis, customer segmentation, machine learning, prompt engineering, Retrieval-Augmented Generation (RAG), and tool-using AI agents.

The system is divided into four parts:

- **Part A:** Data auditing, cleaning, validation, and SQL analysis.
- **Part B:** Customer intelligence, segmentation, and lapse prediction.
- **Part C:** Privacy-aware support-ticket classification and intelligent routing.
- **Part D:** Policy-based RAG and a ReAct refund-support agent.

All datasets are synthetic, and the project uses a local mock LLM implementation. No external API key is required.

## 2. Reproduce

### Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SQLite
- TF-IDF and cosine similarity
- Joblib
- JSON and CSV
- Mock LLM implementation

### Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements file specifies:

- pandas==2.3.3
- numpy==2.3.5
- scikit-learn==1.7.2
- joblib==1.5.2

### Complete End-to-End Execution

From the repository root, install the dependencies and run the complete workflow:

```bash
pip install -r requirements.txt
python run_project.py
```

The main script executes the four project parts in sequence: data generation and cleaning; customer features, RFM segmentation and lapse-model evaluation; ticket redaction, guardrails and routing; and policy chunking, retrieval evaluation and refund-agent scenarios. It stops immediately if a required step fails. The model-comparison stage can take several minutes depending on the computer.

The workflow uses the local mock LLM. It does not require an external API key or network service. Run the command from the repository root so the relative data and output paths resolve correctly.

### Part A Only

To rerun only the data foundation steps from the repository root:

```bash
python generate_data.py
python src/clean_data.py
python src/validate.py
python src/run_sql.py
```

These steps generate synthetic data, clean the datasets, create the SQLite database, validate the data, and execute the seven SQL queries.

### Project Structure

```text
Instamart_Capstone/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
│
├── outputs/
│
├── src/
│
├── sql/
│   └── queries.sql
│
├── state/
│
├── generate_data.py
├── run_project.py
├── mock_llm.py
├── requirements.txt
├── README.md
└── Capstone_Work.ipynb
```

The `src/` directory contains the individual implementation scripts for Parts A–D. Generated datasets, model artifacts, evaluation results, and agent logs are stored in `outputs/`.

## 3. Part A – Data Engineering, Cleaning and SQL Analysis

### Objective

Prepare reliable customer, product, and order datasets for downstream analytics and machine learning.

### Data Cleaning

The synthetic dataset initially contained:

- 1,200 customers
- 40 products
- 13,084 raw order records

The cleaning pipeline performs:

- Customer city and gender standardization.
- Invalid and missing age handling.
- Date conversion.
- Exact duplicate removal.
- Duplicate order-ID resolution.
- Numeric conversion of order amounts.
- Missing completed-order amount calculation.
- Negative delivery-time handling.
- Missing completed-order delivery-time imputation.

### Final Dataset

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |

No pending orders remain in the cleaned dataset.

### Exploratory Data Analysis

The raw orders contained 119 exact duplicate records, 656 pending records, 120 non-numeric order amounts, and 106 negative delivery-time values. After cleaning, the completed-order revenue was ₹29,98,668.80.

The mean completed-order amount was ₹269.91, while the median was ₹135. The distribution was positively skewed, indicating that a smaller number of high-value orders influenced the average.

The largest absolute correlation among the examined numerical variables was between order amount and discount percentage (approximately -0.095). These correlations indicate associations, not causation.

### SQLite and SQL Analysis

The cleaned datasets were stored in `data/instamart.db` with three tables:

- customers
- products
- orders

Seven SQL queries were executed successfully to analyse sales, customers, products, and revenue.

**Validation result:** All 10 database checkpoints passed, including record counts, unique IDs, missing-value checks, and completed-order revenue.

## 4. Part B – Customer Intelligence and Lapse Prediction

### Objective

Identify customer purchasing patterns, create meaningful customer segments, and predict customers who may stop purchasing.

### Historical Feature Engineering

The feature-generation cutoff date was 1 September 2025. Only historical orders before this date were used.

The 12 customer features included:

- Recency
- Frequency
- Monetary value
- Average discount
- Average delivery time
- Cancellation rate
- Favourite product category
- City
- Age
- Gender
- Customer tenure
- Customer ID

The resulting feature dataset contained 1,156 eligible customers and no missing values.

### Lapse Label Creation

The prediction window was September 2025.

A customer was labelled:

- `1` – No completed purchase during September.
- `0` – At least one completed purchase during September.

| Category | Customers |
|---|---:|
| Not lapsed | 713 |
| Lapsed | 443 |
| Total | 1,156 |

The lapse rate was **38.32%**.

### Customer Segmentation

RFM analysis and K-Means clustering were used to identify customer groups.

Four customer segments were identified:

| Segment | Customers |
|---|---:|
| Regular Low-Spend Customers | 436 |
| Loyal Frequent Customers | 419 |
| At-Risk Inactive Customers | 182 |
| Premium High-Value Customers | 138 |

K=4 achieved the highest silhouette score of 0.526 among the tested values from K=2 to K=8.

PCA explained 85.2% of the variance across its first two components.

### Lapse Prediction Models

A stratified 80:20 train-test split was used, with preprocessing fitted only on training data.

Models evaluated included:

- Dummy baseline
- Logistic Regression
- Random Forest
- Oversampled Logistic Regression

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Dummy baseline | 61.64% | — | — | 0.0000 |
| Logistic Regression | 72.84% | 64.44% | 65.17% | 0.6480 |
| Random Forest | 73.71% | 65.91% | 65.17% | 0.6554 |
| Oversampled Logistic Regression | 71.98% | 60.00% | 80.90% | 0.6890 |

The Random Forest model was selected as the final original-data model and saved as:

`outputs/lapse_model.joblib`

Its three most important features were frequency, monetary value, and recency, together accounting for approximately 66.6% of total feature importance.

The retention list is an illustrative prioritisation output. Its premium-customer flag is based on a post-hoc segment definition and should not be interpreted as a fully leakage-free prospective targeting evaluation.

## 5. Part C – Privacy-Aware Support Ticket Triage

### Objective

Automatically classify customer-support tickets, protect personal information, and route tickets to appropriate support teams.

### Privacy and Guardrails

The ticket-processing pipeline:

- Detects empty tickets.
- Flags extremely long tickets.
- Applies a 400-token input limit.
- Redacts email addresses, phone numbers, and order IDs.
- Prevents ineligible inputs from reaching the mock LLM.

The redaction process identified:

| PII Type | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |

The redacted output is stored in `outputs/redacted_tickets.csv`.

### Prompt Comparison

Zero-shot and few-shot prompting were evaluated using the same ticket dataset.

Allowed categories included billing, delivery, technical, and fraud.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 16,576 | 27,340 |
| Tokens per classified ticket | 112.76 | 185.99 |

Few-shot prompting improved classification accuracy but required more tokens.

### Retry and Validation

The pipeline includes:

- JSON output validation.
- Allowed-category checking.
- Retry handling with exponential backoff and jitter.
- Maximum three attempts.
- Invalid-output handling.
- Token-budget enforcement.

Nineteen transient errors were recovered during the evaluation, with no permanent errors.

### Intelligent Routing

Tickets are routed based on classification and customer segment.

- Fraud tickets → Security team.
- Premium customer tickets → Premium support desk.
- Other valid tickets → General support queue.
- Invalid, empty, long, or budget-exhausted tickets → Human review.

The final triage results are stored in:

`outputs/triage_results.csv`

## 6. Part D – Policy RAG and ReAct Refund Agent

### Objective

Build a policy-grounded assistant that retrieves relevant information and uses structured tools to handle refund-related requests safely.

### Policy Knowledge Base

Five policy documents were used:

- Refund policy
- Delivery policy
- Escalation policy
- Payment policy
- Privacy policy

The documents were divided into chunks of 40 words with a 10-word overlap.

This produced 15 chunks.

TF-IDF vectorisation and cosine similarity were used for retrieval. The assistant retrieved the top three relevant chunks for a query.

### Retrieval Evaluation

| Evaluation | 40-word chunks with 10-word overlap | Zero overlap |
|---|---:|---:|
| Total chunks | 15 | 13 |
| Document retrieval | 12/12 | 12/12 |
| Expected facts | 12/12 | 10/12 |
| Correct fallback | 2/2 | 2/2 |

The overlap-based approach improved expected-fact retrieval in the evaluation.

Unanswerable questions were handled using a fallback response rather than inventing policy information.

### Refund Agent and Tools

The ReAct agent uses structured tools:

- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

The refund rules include:

- Only completed orders qualify.
- Delivery exceeding 45 minutes qualifies for a full refund.
- Refunds above ₹1,000 require human approval.
- Refund requests must be within the seven-day policy window.

### Agent Scenario Evaluation

| Scenario | Result |
|---|---|
| S1 | Refund ₹128.25 |
| S2 | Refund ₹135 |
| S3 | Human escalation ₹1,999 |
| S4 | Refund denied – delivery within 45 minutes |
| S5 | Refund denied – cancelled order |
| S6 | Order not found |
| S7 | Maximum-iteration fallback |

The agent supports iteration limits, token limits, time limits, and human handoff.

### Checkpointing and Recovery

Agent execution state is saved under the `state/` directory.

The checkpoint stores the run ID, scenario ID, messages, token usage, step number, and executed tools.

A simulated interruption was recovered successfully. The resumed execution continued without repeating previously completed tool operations.

Agent execution details are stored in:

`outputs/agent_runs.jsonl`

## 7. Limitations and Next Steps

### Limitations

- The project uses synthetic data rather than real customer records.
- The mock LLM does not represent the performance of a production LLM.
- Model performance is dependent on the available dataset and selected features.
- The premium retention list is illustrative and requires a leakage-free prospective evaluation before operational use.
- TF-IDF retrieval may miss relevant information when policy wording differs substantially from a query.
- The agent's tool operations are simulated and are not connected to a real payment or order-management system.
- The evaluation dataset is limited in size and scope.

### Future Improvements

- Evaluate the lapse model using additional real-world or larger synthetic datasets.
- Improve customer-retention targeting through time-based validation.
- Integrate a production LLM with appropriate privacy and cost controls.
- Explore stronger embedding-based retrieval and reranking.
- Add a monitoring dashboard for model and agent performance.
- Integrate with secure order-management and support systems.
- Expand automated evaluation and safety testing.

### Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how data engineering, machine learning, prompt engineering, RAG, and tool-using agents can work together in a practical customer-support workflow.

The project combines customer analytics and predictive modelling with privacy-aware ticket classification and policy-grounded refund assistance, while incorporating validation, safe fallbacks, and execution recovery.