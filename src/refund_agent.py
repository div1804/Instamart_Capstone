
import os
import json
import time
import uuid
import pandas as pd

import mock_llm
from .refund_tools import TOOLS_SPEC, TOOL_MAP


# ============================================================
# 1. CONFIGURATION
# ============================================================

STATE_DIR = "state"
SCENARIO_FILE = "data/raw/agent_scenarios.csv"
ORDERS_FILE = "data/clean/orders.csv"

os.makedirs(STATE_DIR, exist_ok=True)

print("Checkpoint folder ready!")


# ============================================================
# 2. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    llm_calls,
    steps
):

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "llm_calls": llm_calls,
        "steps": steps,
        "step_number": len(steps),
        "tools_executed": [
            step["tool"] for step in steps
        ]
    }

    checkpoint_path = os.path.join(
        STATE_DIR,
        f"{run_id}.json"
    )

    temporary_path = checkpoint_path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            checkpoint,
            file,
            indent=4,
            default=str
        )

    os.replace(temporary_path, checkpoint_path)


# ============================================================
# 3. SCENARIO GENERATION
# ============================================================

def build_scenarios():

    orders = pd.read_csv(ORDERS_FILE)

    required_ids = [
        "ORD-100048",
        "ORD-100067",
        "ORD-100145",
        "ORD-100001",
        "ORD-100020"
    ]

    scenarios = []

    for scenario_id, order_id, expected in [
        ("S1", "ORD-100048", "REFUND"),
        ("S2", "ORD-100067", "REFUND"),
        ("S3", "ORD-100145", "ESCALATE"),
        ("S4", "ORD-100001", "DENIED"),
        ("S5", "ORD-100020", "DENIED")
    ]:

        if order_id not in orders["order_id"].values:
            raise ValueError(
                f"Order {order_id} not found in clean orders."
            )

        scenarios.append({
            "scenario_id": scenario_id,
            "user_message": (
                f"Please check refund eligibility for order {order_id}."
            ),
            "expected_outcome": expected,
            "order_id": order_id
        })

    scenarios.append({
        "scenario_id": "S6",
        "user_message": (
            "Please check refund eligibility for order ORD-999999."
        ),
        "expected_outcome": "NOT_FOUND",
        "order_id": "ORD-999999"
    })

    scenarios.append({
        "scenario_id": "S7",
        "user_message": (
            "[LOOP] Refund order ORD-100048."
        ),
        "expected_outcome": "FALLBACK_MAX_ITERATIONS",
        "order_id": "ORD-100048"
    })

    scenario_df = pd.DataFrame(scenarios)

    os.makedirs(
        os.path.dirname(SCENARIO_FILE),
        exist_ok=True
    )

    scenario_df.to_csv(
        SCENARIO_FILE,
        index=False
    )

    print("Agent scenarios created successfully!")
    print(scenario_df)

    return scenario_df


# ============================================================
# 4. AGENT SYSTEM PROMPT
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are an AGENT for Instamart refund support.

Follow these refund rules carefully:

1. Always check the order using get_order.
2. Use search_policy to retrieve the refund policy.
3. Only completed orders are eligible for refunds.
4. Orders delivered within 45 minutes receive zero refund.
5. Orders delivered after 45 minutes qualify for a full refund.
6. Refund requests must be within the 7-day request window.
7. Refunds above INR 1000 must be escalated to a human.
8. Never automatically issue a refund above INR 1000.
9. Use issue_refund only when the refund is eligible
   and does not exceed INR 1000.
10. If an order is missing, explain that it was not found.
11. Never invent order details or refund amounts.
12. If unsure, stop safely and request human assistance.
"""


# ============================================================
# 5. HUMAN HANDOFF MESSAGE
# ============================================================

def human_handoff(reason, scenario_id, steps):

    return (
        f"{reason}: Agent stopped safely. "
        f"Scenario: {scenario_id}. "
        f"Tools executed: {len(steps)}. "
        "Please review the request and order details manually."
    )


# ============================================================
# 6. AGENT EXECUTION LOOP
# ============================================================

def _continue_agent(
    run_id,
    scenario_id,
    messages,
    steps,
    llm_calls,
    tokens_used,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):

    start_time = time.monotonic()

    final_answer = None
    status = None

    while True:

        # TIME LIMIT CHECK
        if time.monotonic() - start_time >= time_limit_seconds:

            status = "FALLBACK_TIMEOUT"

            final_answer = human_handoff(
                status,
                scenario_id,
                steps
            )

            break

        # MAXIMUM ITERATION CHECK
        if llm_calls >= max_iterations:

            status = "FALLBACK_MAX_ITERATIONS"

            final_answer = human_handoff(
                status,
                scenario_id,
                steps
            )

            break

        # CALL MOCK LLM
        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1

        tokens_used += response.usage.total_tokens

        assistant_message = response.choices[0].message

        tool_calls = assistant_message.tool_calls

        # TOKEN LIMIT CHECK
        # Do not execute requested tools after exceeding the limit.

        if tokens_used > token_limit:

            status = "FALLBACK_TOKEN_LIMIT"

            final_answer = human_handoff(
                status,
                scenario_id,
                steps
            )

            break

        # CHECK FINAL ANSWER
        if not tool_calls:

            final_answer = assistant_message.content or ""

            if ":" in final_answer:
                status = final_answer.split(":")[0].strip()
            else:
                status = "COMPLETED"

            break

        # SAVE ASSISTANT TOOL CALLS IN MESSAGE HISTORY
        assistant_tool_calls = []

        for tc in tool_calls:

            assistant_tool_calls.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # EXECUTE TOOLS
        for tc in tool_calls:

            tool_name = tc.name

            arguments = tc.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                raise ValueError(
                    f"Unknown tool requested: {tool_name}"
                )

            tool_function = TOOL_MAP[tool_name]

            observation = tool_function(**arguments)

            observation_json = json.dumps(
                observation,
                default=str
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": observation_json
            })

            steps.append({
                "tool": tool_name,
                "args": arguments,
                "observation": observation
            })

            # CHECKPOINT AFTER EVERY TOOL EXECUTION
            save_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                tokens_used=tokens_used,
                llm_calls=llm_calls,
                steps=steps
            )

            # SIMULATE CRASH AFTER CHECKPOINT IS SAVED
            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):

                raise RuntimeError(
                    f"SIMULATED_CRASH: {run_id}"
                )

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


# ============================================================
# 7. RUN A NEW AGENT
# ============================================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):

    run_id = run_id or (
        f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    )

    messages = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    return _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=[],
        llm_calls=0,
        tokens_used=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


# ============================================================
# 8. RESUME AGENT FROM CHECKPOINT
# ============================================================

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    checkpoint_path = os.path.join(
        STATE_DIR,
        f"{run_id}.json"
    )

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with open(
        checkpoint_path,
        "r",
        encoding="utf-8"
    ) as file:

        checkpoint = json.load(file)

    print("Checkpoint loaded successfully!")
    print("Run ID:", checkpoint["run_id"])
    print("Scenario:", checkpoint["scenario_id"])
    print("Previously executed tools:",
          checkpoint["tools_executed"])

    return _continue_agent(
        run_id=checkpoint["run_id"],
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        steps=checkpoint["steps"],
        llm_calls=checkpoint["llm_calls"],
        tokens_used=checkpoint["tokens_used"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


print("Refund agent module loaded successfully!")
