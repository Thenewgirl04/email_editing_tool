import json

import yaml

from config.config import EDGE_CASE_ID_THRESHOLD, EVAL_MODELS, JUDGE_MODEL
from generators.generate_response import GenerateEmail

with open("prompts/evaluation_task_prompts.yaml", "r", encoding="utf-8") as f:
    task_prompts = yaml.safe_load(f)

with open("prompts/evaluation_prompts.yaml", "r", encoding="utf-8") as f:
    judge_prompts = yaml.safe_load(f)

JUDGE_METRICS = ["faithfulness_judge", "completeness_judge", "relevance_judge"]


def parse_judge_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "rating": 0,
            "explanation": f"Could not parse judge response: {text[:200]}",
        }


def data_type_for_email(task: str, email_id: int, include_edge_cases: bool) -> str:
    if include_edge_cases and email_id > EDGE_CASE_ID_THRESHOLD:
        return "edge_case"
    return "baseline"


def evaluate(email: dict, task: str, tone_style: str = "professional", include_edge_cases: bool = False) -> list[dict]:
    selected_text = email["content"]
    email_id = email.get("id")
    data_type = data_type_for_email(task, email_id, include_edge_cases)
    results = []

    for model in EVAL_MODELS:
        response_generator = GenerateEmail(model=model, prompts=task_prompts)

        prompt_kwargs = {"selected_text": selected_text}
        if task == "tone":
            prompt_kwargs["tone_style"] = tone_style
            system_prompt = response_generator.get_prompt("tone", prompt_type="system")
            user_prompt = response_generator.get_prompt("tone", prompt_type="user", **prompt_kwargs)
        else:
            system_prompt = response_generator.get_prompt(task, prompt_type="system", **prompt_kwargs)
            user_prompt = response_generator.get_prompt(task, prompt_type="user", **prompt_kwargs)

        model_response = response_generator.send_prompt(user_prompt, system_prompt)

        evaluations = {}
        for metric in JUDGE_METRICS:
            judge = GenerateEmail(model=JUDGE_MODEL, prompts=judge_prompts)
            judge_system = judge.get_prompt(metric, prompt_type="system")
            judge_user = judge.get_prompt(
                metric,
                prompt_type="user",
                selected_text=selected_text,
                model_response=model_response,
            )
            judge_verdict = judge.send_prompt(judge_user, judge_system)
            evaluations[metric] = parse_judge_response(judge_verdict)

        results.append(
            {
                "email_id": email_id,
                "data_type": data_type,
                "task": task,
                "input": selected_text,
                "model_output": model_response,
                "model": model,
                "faithfulness_rating": evaluations["faithfulness_judge"]["rating"],
                "faithfulness_explanation": evaluations["faithfulness_judge"]["explanation"],
                "completeness_rating": evaluations["completeness_judge"]["rating"],
                "completeness_explanation": evaluations["completeness_judge"]["explanation"],
                "relevance_rating": evaluations["relevance_judge"]["rating"],
                "relevance_explanation": evaluations["relevance_judge"]["explanation"],
            }
        )

    return results
