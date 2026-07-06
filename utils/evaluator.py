#Generate a response
#Evaluate the response
import yaml
import json

from generators.generate_response import GenerateEmail

with open("prompts/evaluation_task_prompts.yaml", "r") as f:
    task_prompts = yaml.safe_load(f)

with open("prompts/evaluation_prompts.yaml", "r") as f:
    judge_prompts = yaml.safe_load(f)


def evaluate(email, task):
    models = ["gpt-4o-mini", "gpt-4.1"]
    selected_text = email["content"]
    judge_metrics = ['faithfulness_judge', 'completeness_judge', 'relevance_judge']
    results = []

    for model in models:
        response_generator = GenerateEmail(model=model, prompts=task_prompts)

        prompt_kwargs = {'selected_text': selected_text}
        if task == 'tone':
            prompt_kwargs['tone_style'] ="professional"
        system_prompt = response_generator.get_prompt(task,prompt_type='system')
        user_prompt = response_generator.get_prompt(task, prompt_type='user', **prompt_kwargs)
        model_response = response_generator.send_prompt(user_prompt, system_prompt)

        evaluations = {}
        for metric in judge_metrics:
            judge = GenerateEmail(model="gpt-4.1", prompts=judge_prompts)
            judge_system = judge.get_prompt(metric, prompt_type="system")
            judge_user = judge.get_prompt(metric, prompt_type='user', selected_text=selected_text, model_response=model_response)

            judge_verdict = judge.send_prompt(judge_user, judge_system)
            print(f"Judge verdict for {metric}:")
            print(repr(judge_verdict))  # Print raw response
            print("---")
            evaluations[metric] = json.loads(judge_verdict)
        
        results.append({
            'email_id': email.get('id'),
            'input': selected_text,
            'model_output': model_response,
            'model': model,
            'faithfulness_rating': evaluations['faithfulness_judge']['rating'],
            'faithfulness_explanation': evaluations['faithfulness_judge']['explanation'],
            'completeness_rating': evaluations['completeness_judge']['rating'],
            'completeness_explanation': evaluations['completeness_judge']['explanation'],
            'relevance_rating': evaluations['relevance_judge']['rating'],
            'relevance_explanation': evaluations['relevance_judge']['explanation'],
        })
    
    return results



        


