"""Run the Instamart Customer Intelligence & Support Copilot end to end.

Run this file from the repository root after installing requirements:
    python run_project.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("Part A - Generate synthetic data", "generate_data.py"),
    ("Part A - Clean data and build SQLite database", "src/clean_data.py"),
    ("Part A - Validate cleaned data", "src/validate.py"),
    ("Part A - Execute SQL analysis", "src/run_sql.py"),
    ("Part B - Build historical customer features", "src/build_customer_features.py"),
    ("Part B - Build lapse cohort", "src/build_lapse_dataset.py"),
    ("Part B - Build RFM table", "src/build_rfm.py"),
    ("Part B - Evaluate K-Means values", "src/kmeans_analysis.py"),
    ("Part B - Profile segments and run PCA", "src/segment_profiling.py"),
    ("Part B - Save customer segments", "src/save_customer_segments.py"),
    ("Part B - Build lapse prediction dataset", "src/prepare_lapse_dataset.py"),
    ("Part B - Check preprocessing", "src/preprocess_lapse_data.py"),
    ("Part B - Check train-test split", "src/split_lapse_data.py"),
    ("Part B - Evaluate baseline model", "src/baseline_model.py"),
    ("Part B - Evaluate Logistic Regression", "src/logistic_regression_model.py"),
    ("Part B - Evaluate Random Forest", "src/random_forest_model.py"),
    ("Part B - Evaluate oversampled Logistic Regression", "src/oversampling_model.py"),
    ("Part B - Save and verify final lapse model", "src/save_lapse_model.py"),
    ("Part B - Generate retention list", "src/premium_retention_list.py"),
    ("Part C - Redact ticket PII", "src/redact_tickets.py"),
    ("Part C - Check ticket input guardrails", "src/ticket_guardrails.py"),
    ("Part C - Compare zero-shot and few-shot prompts", "src/compare_prompts.py"),
    ("Part C - Generate final ticket routing", "src/triage_final.py"),
    ("Part D - Build policy chunks", "src/policy_chunking.py"),
    ("Part D - Evaluate policy retrieval", "src/evaluate_policy.py"),
    ("Part D - Run policy assistant example", "src/policy_assistant.py"),
    ("Part D - Run refund-agent scenarios and recovery checks", "src/run_refund_agent.py"),
]


def main():
    print("=" * 72)
    print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("END-TO-END PROJECT EXECUTION")
    print("=" * 72)

    for index, (label, script) in enumerate(STAGES, start=1):
        script_path = ROOT / script
        if not script_path.is_file():
            raise FileNotFoundError(f"Required project script not found: {script}")

        print(f"\n[{index}/{len(STAGES)}] {label}", flush=True)
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        required_paths = [str(ROOT), str(ROOT / "src")]
        if existing_pythonpath:
            required_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(required_paths)
        # Limit numerical-library threads to avoid CPU oversubscription on laptops.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[variable] = "1"

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            env=env,
            check=True,
        )

    print("\n" + "=" * 72)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("Review the generated files in data/, outputs/, and state/.")
    print("=" * 72)


if __name__ == "__main__":
    main()
