from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import json

load_dotenv()

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

with open("judge.yaml","r") as j:
    judge_prompts = yaml.safe_load(j)

class GenerateEmail():
    def __init__(self, model: str):
        # initialize client once
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self, messages):
        # TODO: implement this function to call ChatCompletions
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        return response.choices[0].message.content
    
    def get_prompt(self, prompt_name, prompt_type='user', **kwargs):
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)
    

    def send_prompt(self, user_prompt: str, system_msg):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)
    
    def generate(self, action: str, selected_text: str = "Hello world") -> list:
        # TODO: implement your backend logic with this method. Skeleton code is provided below.
        if action == "lengthen":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('lengthen', prompt_type='system', **args)
            user_prompt = self.get_prompt('lengthen', **args)
            model_response = self.send_prompt(user_prompt,system_prompt)
            return model_response


        if action == "shorten":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('shorten', prompt_type='system', **args)
            user_prompt = self.get_prompt('shorten', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        if action == "professional":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('professional', prompt_type='system', **args)
            user_prompt = self.get_prompt('professional', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        if action == "friendly":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('friendly', prompt_type='system', **args)
            user_prompt = self.get_prompt('friendly', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response

        if action == "sympathetic":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('sympathetic', prompt_type='system', **args)
            user_prompt = self.get_prompt('sympathetic', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
            return model_response


    def parse_json(text: str):
        return json.loads(text)

    def normalize_judge_output(self,parsed: dict, judge_name: str=None):
        normalized = {
            "rating": parsed.get("rating"),
            "explanation": parsed.get("explanation")
        }
        if judge_name == "word_count_judge":
            normalized["word_count_change_percent"] = parsed.get("word_count_change_percent")
        return normalized


    def evaluate(self, selected_text, model_response , original_word_count=None, edited_word_count=None, word_count_change_percent=None, word_count_change_direction=None):
        results = {}
        for judge in judge_prompts:

            format_args = {
                "selected_text": selected_text,
                "model_response": model_response
            }

            if original_word_count is not None:
                format_args["original_word_count"] = original_word_count
            if edited_word_count is not None:
                format_args["edited_word_count"] = edited_word_count
            if word_count_change_percent is not None:
                format_args["word_count_change_percent"] = word_count_change_percent
            if word_count_change_direction is not None:
                format_args["word_count_change_direction"] = word_count_change_direction

            judge_response = self.send_prompt(
                user_prompt=judge_prompts[judge]["user"]["en"].format(**format_args),
                system_msg=judge_prompts[judge]["system"]["en"]
            )

            # Be robust to malformed JSON from the judge model
            try:
                parsed = json.loads(judge_response)
            except json.JSONDecodeError:
                parsed = {
                    "rating": 0,
                    "explanation": f"Failed to parse judge response as JSON. Raw output was: {judge_response}"
                }

            results[judge] = self.normalize_judge_output(parsed, judge_name=judge)

        return results

