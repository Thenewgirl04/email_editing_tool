from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import json

load_dotenv()

with open("prompts/prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

class GenerateEmail():
    def __init__(self, model: str, prompts: dict = None):
        # initialize client once
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model
        self.prompts = prompts

    def _call_api(self, messages):
        # TODO: implement this function to call ChatCompletions
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        return response.choices[0].message.content

    def get_prompt(self, prompt_name, prompt_type='user', **kwargs):
        prompts_to_use = self.prompts if self.prompts is not None else globals().get('prompts')
        template = prompts_to_use[prompt_name][prompt_type]

        print(f"Template for {prompt_name}: {prompt_type}:")
        print(repr(template))
        print("---")

        if isinstance(template,dict):
            template = template['en']

        if kwargs:
            return template.format(**kwargs)
        else:
            return template

    def send_prompt(self, user_prompt: str, system_msg):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages)

    def generate(self, action: str, selected_text: str = "Hello world") -> str | None:
        # TODO: implement your backend logic with this method. Skeleton code is provided below.
        if action == "lengthen":
            args = {
                "selected_text": selected_text
            }
            system_prompt = self.get_prompt('lengthen', prompt_type='system', **args)
            user_prompt = self.get_prompt('lengthen', **args)
            model_response = self.send_prompt(user_prompt, system_prompt)
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
        return 'Action not Valid'


