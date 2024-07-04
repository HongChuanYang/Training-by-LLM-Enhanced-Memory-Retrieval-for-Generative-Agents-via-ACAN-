import os


def prompt_generate(prompt_input, prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    for count, i in enumerate(prompt_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    return prompt

