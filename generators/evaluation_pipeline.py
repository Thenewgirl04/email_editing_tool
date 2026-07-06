##Load the emails in the data set
##Generate the response from both models for each three tasks
##Load the evaluation prompts and use 4.1 as the model to evaluate 4o mini and 4.1

# For each dataset:
#     Load .jsonl file
#     For each email:
#         Call Model A with task-specific prompt
#         Call Model B with task-specific prompt
#         Call Judge on both outputs
#         Store results
#     Save results to CSV

import json
import csv
from utils.evaluator import evaluate

class EvaluationPipeline:
    def __init__(self, tasks, datasets):
        self.tasks =  tasks
        self.datasets = datasets

    def pipeline(self):
        for task, dataset_path in zip(self.tasks, self.datasets):
            all_results = []

            with open(dataset_path, 'r') as f:
                for line in f:
                    email = json.loads(line)
                    results = evaluate(email, task)
                    all_results.extend(results)


            csv_path = f'results/{task}_results.csv'
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['email_id', 'input', 'model_output', 'model',
                              'faithfulness_rating', 'faithfulness_explanation',
                              'completeness_rating', 'completeness_explanation',
                              'relevance_rating', 'relevance_explanation']

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(all_results)

        print(f"Saved {len(all_results)} rows to {csv_path}")






