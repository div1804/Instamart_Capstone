"""Run the seven offline refund-agent scenarios and recovery checks."""

import json
import os

from .refund_agent import build_scenarios, run_agent, resume_agent


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("state", exist_ok=True)

    scenarios = build_scenarios()
    results = []

    for row in scenarios.to_dict(orient="records"):
        scenario_id = row["scenario_id"]
        message = row["user_message"]

        if scenario_id == "S1":
            # Demonstrate recovery from a crash after two completed tool calls.
            run_id = "S1_RUNNER_CRASH_RESUME"
            try:
                run_agent(
                    message,
                    scenario_id,
                    run_id=run_id,
                    crash_after_tools=2,
                )
            except RuntimeError as exc:
                if "SIMULATED_CRASH" not in str(exc):
                    raise
            result = resume_agent(run_id)
        else:
            result = run_agent(message, scenario_id)

        results.append(result)
        print(
            f"{scenario_id}: {result['status']} | "
            f"tool calls={len(result['steps'])} | "
            f"tokens={result['tokens_used']}"
        )

    output_path = "outputs/agent_runs.jsonl"
    with open(output_path, "w", encoding="utf-8") as file:
        for result in results:
            record = {
                "scenario_id": result["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Explicitly verify the two safety fallbacks documented in the project.
    token_fallback = run_agent(
        "Please check refund eligibility for order ORD-100048.",
        "TOKEN_LIMIT_CHECK",
        token_limit=200,
    )
    timeout_fallback = run_agent(
        "Please check refund eligibility for order ORD-100048.",
        "TIMEOUT_CHECK",
        time_limit_seconds=0,
    )

    if token_fallback["status"] != "FALLBACK_TOKEN_LIMIT":
        raise AssertionError("Token-limit fallback did not trigger as expected.")
    if timeout_fallback["status"] != "FALLBACK_TIMEOUT":
        raise AssertionError("Timeout fallback did not trigger as expected.")

    print("Token fallback:", token_fallback["status"])
    print("Timeout fallback:", timeout_fallback["status"])
    print("Agent results saved to:", output_path)


if __name__ == "__main__":
    main()
