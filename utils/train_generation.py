from utils.text_generation import generate
from utils.utils import parse_json_from_content
import json
import re
from utils.prompt_generation import prompt_generate


def parse_score(llm_output):
    try:
        scores = json.loads(llm_output)
        action_a_score = int(scores.get('memory_a_score', 5))
        action_b_score = int(scores.get('memory_b_score', 5))
        return action_a_score, action_b_score
    except:
        # If there's any error in JSON parsing or int conversion
        try:
            match = re.search(r'\b\d+\b', llm_output)
            if match:
                action_a_score = int(match.group(0))
                match = re.search(r'\b\d+\b', llm_output[match.end():])
                if match:
                    action_b_score = int(match.group(0))
                    return action_a_score, action_b_score
        except:
            # If there's any error in regex matching or int conversion
            pass

    return 5, 5


def generate_score(base_prompt, y_pre, y_true):
    prompt_path = "prompt/generate_loss.txt"
    prompt_input = []
    prompt_input.append(y_pre)
    prompt_input.append(y_true)
    prompt_input.append(base_prompt)
    prompt = prompt_generate(prompt_input, prompt_path)
    llm_output, token = generate(prompt)
    score_pre, score_true = parse_score(llm_output)
    output_score = (int(score_pre)-int(score_true))/10
    return output_score, token


    # based agent memories generate agent action
    
# def judge_action():
    