# Instamart – AI-Powered Customer Intelligence and Support Automation

## 1. Project Overview

Instamart is an AI-powered data analytics and automation project developed using Python, Machine Learning, SQL, and Retrieval-Augmented Generation (RAG).

The project demonstrates how customer data, order information, support tickets, and policy documents can be used to support business decisions and automate customer support workflows.

The project uses fictional synthetic datasets for demonstration and testing.

## 2. Project Objectives

- Clean and validate customer, order, and product datasets.
- Store and analyse structured data using SQLite and SQL.
- Identify customer segments using RFM analysis.
- Build a machine learning model for customer lapse prediction.
- Automate support ticket classification with guardrails.
- Build a policy-based RAG assistant.
- Develop refund tools and an AI agent with safety controls.
- Test agent execution, checkpointing, and recovery.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SQLite and SQL
- TF-IDF and cosine similarity
- Retrieval-Augmented Generation (RAG)
- JSON and JSONL
- Jupyter Notebook
- Mock LLM for offline agent testing

## 4. Project Structure

```text
Instamart_Capstone/
│
├── Capstone_Work.ipynb
├── generate_data.py
├── mock_llm.py
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
│
├── outputs/
│   ├── customer_segments.csv
│   ├── lapse_dataset.csv
│   ├── lapse_model.joblib
│   ├── premium_retention_list.csv
│   ├── triage_results.csv
│   ├── policy_chunks.json
│   ├── policy_evaluation.csv
│   ├── refund_ledger.jsonl
│   └── agent_runs.jsonl
│
├── sql/
│   └── queries.sql
│
├── src/
│   ├── clean_data.py
│   ├── customer_segmentation.py
│   ├── policy_assistant.py
│   ├── policy_retrieval.py
│   ├── refund_agent.py
│   ├── refund_tools.py
│   └── other project modules
│
└── state/
    └── agent checkpoint files
```

## 5. Project Implementation

### Part A – Data Foundation

- Generated synthetic customer, order, product, and support ticket datasets.
- Cleaned and validated datasets.
- Stored structured data in SQLite.
- Performed SQL queries and exploratory data analysis.

### Part B – Customer Intelligence

- Performed RFM analysis.
- Identified four customer segments.
- Prepared customer segment profiles.
- Developed a customer lapse prediction workflow.
- Selected a Random Forest model.
- Generated a premium customer retention list.

### Part C – Support Ticket Triage

- Implemented support ticket data redaction.
- Compared zero-shot and few-shot prompting.
- Implemented ticket classification and validation.
- Added guardrails and retry handling.
- Tested the triage workflow.

### Part D – Policy Assistant and AI Agent

- Created policy document chunks with overlap.
- Implemented TF-IDF-based document retrieval.
- Developed a RAG-based policy assistant.
- Evaluated retrieval and answer quality.
- Implemented refund-related tools.
- Developed an AI agent using tool calling.
- Implemented token, iteration, and timeout limits.
- Implemented checkpointing and resume functionality.
- Compared ReAct and Plan-and-Execute approaches.

## 6. Safety and Reliability

The project includes:

- Synthetic data for demonstration.
- Customer information redaction.
- Policy-based refund decisions.
- Tool execution controls.
- Token and iteration limits.
- Timeout handling.
- Checkpoint-based recovery.
- Execution logs for auditing.

The mock LLM is used for offline testing. No external LLM API key is required for the demonstrated mock-agent workflow.

## 7. Running the Project

1. Clone or download the project repository.
2. Open the project folder.
3. Install the required Python dependencies listed in `requirements.txt`.
4. Open `Capstone_Work.ipynb` using Jupyter Notebook.
5. Run the notebook cells in the required order.

Some modules can also be executed independently from the project root.

## 8. Limitations and Future Improvements

- Use real-time business data with appropriate authorization.
- Integrate a production-grade LLM API.
- Improve model performance with larger validated datasets.
- Develop a web-based dashboard.
- Add authentication and role-based access control.
- Deploy the application to a cloud platform.

## 9. Conclusion

This project demonstrates the practical application of data analytics, machine learning, SQL, RAG, and AI agent workflows in an e-commerce customer support environment.

It combines customer intelligence with policy-based automation while emphasizing reliability, safety, and traceability.

---

**Project:** Instamart – AI-Powered Customer Intelligence and Support Automation  
**Environment:** Python and Jupyter Notebook  
**Data:** Fictional synthetic datasets