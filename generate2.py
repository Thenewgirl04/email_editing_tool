from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import json
import sys


load_dotenv()

with open("gen_prompts.yaml",'r') as f:
    prompts = yaml.safe_load(f)

class Generate_data:
    def __init__(self, model):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self,messages):
        response = self.client.chat.completions.create(
            model=self.deployment_name, 
            messages=messages
        )
        return response.choices[0].message.content
    
    def get_prompt(self, prompt_name,prompt_type="user",**kwargs):
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)
    
    def send_prompt(self, user_prompt: str, system_msg):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)
    
    def generate(self,  prompt_name, system_msg,output_path,n_samples: int = 10, **prompt_vars):
        user_prompt = self.get_prompt(prompt_name,n_samples=n_samples,**prompt_vars)
        raw_output = self.send_prompt(user_prompt, system_msg)
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError ("Model output is not valid JSON")
        
        if isinstance(parsed, dict):
            parsed = [parsed]

        with open(output_path, "a", encoding="utf-8") as f:
            for record in parsed:
                f.write(json.dumps(record) + "\n")

        return len(parsed)


generator = Generate_data(model="gpt-4.1")


tasks = [
    ("long_input", "dataset_copy/lengthen.jsonl"),
    ("short_input", "dataset_copy/shorten.jsonl"),
    ("tone_input", "dataset_copy/tone.jsonl"),
]

total_added = 0
for prompt_name, output_path in tasks:
    added = generator.generate(
        prompt_name=prompt_name,
        system_msg=prompts[prompt_name]["system"],
        output_path=output_path,
    )
    total_added += added
    print(f"Added {added} samples to {output_path}")

print(f"Total: {total_added} new samples")
    
 