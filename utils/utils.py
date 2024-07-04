import os
import re
import json

def get_time_phase(day_global_time):
    if day_global_time < 3: # 6 - 9 morning
        return "morning" + " {}:00 am".format(6 + day_global_time) + ", it's nonworking time"
    elif 3 <= day_global_time < 6:  # 9 - 12 
        return " {}:00 am".format(6 + day_global_time) + ", it's working time"
    elif 6 <= day_global_time < 7:  # 12 - 13 
        return " {}:00 pm".format(6 + day_global_time) + ", it's lunch time"
    elif 7 <= day_global_time < 11:  # 13 - 16
        return " {}:00 pm".format(6 + day_global_time) + ", it's working time"
    elif 11 <= day_global_time < 15:  # 17 - 20
        return " {}:00 pm".format(6 + day_global_time) + ", it's nonworking time"
    elif day_global_time == 15:  # 21 sleep
        return " {}:00 pm".format(6 + day_global_time) + ", it's sleeping time"
    else:  # 意外情况``
        return " {}:00 pm".format(6 + day_global_time) + ", it's other time"
    
    
def get_pure_time_phase(day_global_time):
    if day_global_time < 6: # am
        return " {}:00 am".format(6 + day_global_time)
    else:  # pm
        return " {}:00 pm".format(6 + day_global_time)
    

def reverse_get_pure_time_phase(pure_time_phase):
    if "am" in pure_time_phase:
        return int(pure_time_phase.split(":")[0]) - 6
    else:
        return int(pure_time_phase.split(":")[0]) - 6
    

def chat_history_to_str(chat_history):
# 创建一个空字符串用于存储最终的对话文本
    chat_str = ""
    # 遍历chat_history中的每个子列表
    for entry in chat_history:
        name, response = entry
        # 将每个对话格式化并添加到chat_str中
        chat_str += f"{name}: {response}\n"
    return chat_str

def find_location_in_response(response, locations_name):
    # Iterate through each location in the list
    for index, location in enumerate(locations_name):
        # Check if the location name is mentioned in the LLM's response
        if re.search(re.escape(location), response, re.IGNORECASE):
            return index  # Return the index of the location in the list
    
    # If no location matches, return False
    return False

def parse_json_from_content(content):
    normalized_json = {}
    # 第一步：搜索和处理Importance score
    search_content = content[-30:] 
    score_pattern = re.compile(r'"?importance[\s_]*scores?"?\s*[:=]\s*(\d+)', re.IGNORECASE)
    score_match = score_pattern.search(search_content)
    if score_match:
        # 将匹配到的数字转化为整数，并存入normalized_json
        normalized_json['importance_score'] = int(score_match.group(1))
        # 删除"Importance score"及其后的所有内容
        content = content[:len(content) - 30 + score_match.start()].strip()
    # 第二步：从剩余内容中提取并删除plan或action
    cleaned_content = re.sub(r'^\{\s*"\w+"\s*:\s*"', '', content)
    # 删除尾部的可能存在的JSON格式字符
    cleaned_content = re.sub(r'"\s*\}\s*$', '', cleaned_content)
    # 额外清除内部可能出现的尾部逗号，这通常是因为原JSON中有多个键值对
    cleaned_content = re.sub(r'"\s*,\s*\w+\s*:\s*.*$', '', cleaned_content)
    normalized_json['content'] = cleaned_content.strip()
    return normalized_json

def parse_chat(content):
    # 初始化返回的字典
    parsed_output = {
        'response': '',
        'continue_the_conversation': True  # 默认设置为 True
    }

    # 检查内容中是否包含 'False' 来决定是否继续对话
    if re.search(r'\bFalse\b', content, re.IGNORECASE):
        parsed_output['continue_the_conversation'] = False
    
    # 从内容中去掉 'continue the conversation' 和其值
    content_without_continue = re.sub(r'"continue[\s_\-]?the[\s_\-]?conversation"\s*:\s*(true|false)\s*,?', '', content, flags=re.IGNORECASE)
    
    # 使用正则表达式提取 'response'
    response_pattern = re.compile(r'"response"\s*:\s*"([^"]+)"', re.IGNORECASE)
    response_match = response_pattern.search(content_without_continue)
    if response_match:
        parsed_output['response'] = response_match.group(1)

    return parsed_output