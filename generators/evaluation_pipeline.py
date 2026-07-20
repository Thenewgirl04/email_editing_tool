import csv
from pathlib import Path

from config.config import TASKS, results_dir
from utils.evaluator import evaluate
from utils.jsonl_loader import load_jsonl

RESULT_FIELDS = [
    "email_id",
    "data_type",
    "task",
    "input",
    "model_output",
    "model",
    "faithfulness_rating",
    "faithfulness_explanation",
    "completeness_rating",
    "completeness_explanation",
    "relevance_rating",
    "relevance_explanation",
]


class EvaluationPipeline:
    def __init__(self, tasks=None, dataset_paths=None):
        self.tasks = tasks or TASKS
        self.dataset_paths = dataset_paths

    def pipeline(
        self,
        include_edge_cases: bool = False,
        tone_style: str = "professional",
        progress_callback=None,
    ) -> dict[str, Path]:
        output_root = results_dir(include_edge_cases)
        output_root.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        dataset_paths = self.dataset_paths
        if dataset_paths is None:
            from config.config import dataset_paths as default_dataset_paths

            dataset_paths = default_dataset_paths(include_edge_cases)

        task_emails = {}
        for task, dataset_path in zip(self.tasks, dataset_paths):
            task_emails[task] = load_jsonl(dataset_path)

        total_steps = sum(len(emails) for emails in task_emails.values())
        completed = 0

        for task, dataset_path in zip(self.tasks, dataset_paths):
            all_results = []
            emails = task_emails[task]

            for email in emails:
                results = evaluate(
                    email,
                    task,
                    tone_style=tone_style,
                    include_edge_cases=include_edge_cases,
                )
                all_results.extend(results)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_steps, task, email.get("id"))

            csv_path = output_root / f"{task}_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(all_results)

            saved_paths[task] = csv_path

        return saved_paths
