# 1
from utils.text_generation import generate, generate_embedding
from utils.prompt_generation import prompt_generate
from utils.utils import get_time_phase, get_pure_time_phase, chat_history_to_str, find_location_in_response, parse_json_from_content, parse_chat
from memory.base_memory import AssociativeMemory
from memory.acan_memory import AcanMemory
import re
import os


class Agent:
    def __init__(self, name, detail_description, all_locations, memory_type):
        self.name = name
        self.detail_description = detail_description
        self.brief_description = f"Age: {detail_description['Age']}, Gender: {detail_description['Gender']}, Occupation: {detail_description['Occupation']}"
        self.all_locations = all_locations
        self.workplace = "anywhere"
        for name, location in self.all_locations.items():     
            if self.name in location.agents and location.type == "workplace":
                self.workplace = name
        self.home = self.name + " Home"
        self.location = self.home
        self.plans = ""
        self.prompt_meta = '### start fresh \n### I want you to play the role of {}, with the following description: {} \nAccording to the instructions provided below, respond in the voice of the assigned role.'.format(self.name, self.brief_description + ', Brief description: {} Duties: {} Interactions: {} Frequent Locations: {}'.format(self.detail_description['Description'], self.detail_description['Duties'], self.detail_description['Interactions'], self.detail_description['Frequent Locations'])) + '\n### Instructions:\n{}\n### Response:<>'
        self.current_action = "woke up"
        self.current_state = ""
        self.working_state = False
        self.talk_dict = {}
        self.peoples_nearby = {}
        self.nearby_people = '' 
        
        agent_mem_path = os.path.join('memory/agent_memory', self.name)
        self.a_mem = AssociativeMemory(agent_mem_path, self.location) if memory_type == 'base' else AcanMemory(agent_mem_path, self.location)
        self.plans =  self.a_mem.get_last_memory('thought').description if self.a_mem.get_last_memory('thought') else self.plans
        self.current_action = self.a_mem.get_last_memory('event').description if self.a_mem.get_last_memory('event') else self.current_action
        self.location = self.a_mem.location
        
        
    def __repr__(self):
        return f"Agent({self.name}, {self.brief_description}, {self.location})"
        
    def get_current_state(self):
        location = self.location if self.location != self.home else "your home"
        location_description = self.all_locations[self.location].description
        time_phase = get_time_phase(self.global_time)
        work_phase = "must working in your workplace: {}".format(self.workplace) if self.working_state else "you can do anything or go anywhere you like"
        if self.plans == "":
            return 'You just {} in {}. The description of this place was: {} Your have no plan about what to do next. Now the time was: {}. In this time you {}'.format(self.current_action, location, location_description, time_phase, work_phase)
        else:
            return 'You just {} in {}. The description of this place was: {} Your last plans was: {} Now the time was: {}. In this time you {}'.format(self.current_action, location, location_description, self.plans, time_phase, work_phase)

    def plan(self, agents, global_time, global_day):
        # 根据时间段修改提示
        self.global_day = global_day + 1
        self.global_time = global_time
        self.peoples_nearby = {name: agent for name, agent in agents.items() if agent.location == self.location and name != self.name}
        self.talk_dict = {}
        nearby_people = ''
        if self.peoples_nearby:
            nearby_people = 'There are peopole is in the same place with you. And the description of these target people was: '
            for name, agent in self.peoples_nearby.items():
                nearby_people += f'{name} : {agent.brief_description}; '
            talk_list = self.decide_to_talk(self.peoples_nearby)
            self.talk_dict = {name: True for name in talk_list}
            if any(self.talk_dict.values()):
                self.plans = 'Have a conversation with ' + ' and '.join(talk_list) + '.'
                poignancy = 5
                self.a_mem.add_thought(self.global_day, self.global_time, self.plans, poignancy, self.location, self.location)
                return
            #     self.talk_action(peoples_nearby, talk_dict)
        if nearby_people:
            nearby_people += ' You already decided to do not have chat with any of these people.'
        self.nearby_people = nearby_people
        prompt_path = "prompt/plan.txt"
        prompt_input = []
        self.current_state = self.get_current_state()
        prompt_input.append(self.current_state)
        prompt_input.append(self.nearby_people)
        query = self.current_state + '. Now you are making a plan for this time at {}'.format(self.location)
        relevant_memory, memory_state = self.a_mem.extract(query)
                
        prompt_no_mem = prompt_generate(prompt_input, prompt_path)
        prompt = prompt_no_mem.replace(f"!<INPUT mem>!", relevant_memory)
        llm_output, token  = generate(self.prompt_meta.format(prompt))
        json_output = parse_json_from_content(llm_output)
        self.plans = json_output.get('content') if json_output.get('content') else llm_output           
        if json_output.get('importance_score'):
            # Convert the matched importance score to an integer and check its range
            importance_score = int(json_output.get('importance_score'))
            if importance_score < 0 or importance_score > 9:
                importance_score = 5 # Set default value if out of range
        else:
            importance_score = 5 # Set default value if no match was found    
        self.a_mem.add_thought(self.global_day, self.global_time, self.plans, importance_score, self.location, self.location)
        self.a_mem.query_with_mem.append({'query': query, 'memory_state': memory_state, 'action':llm_output, 'action_type':'thought', 'prompt':self.prompt_meta.format(prompt_no_mem), 'base_memory':relevant_memory})
        
    
    def test_talk(self, name, content):
        chat_dict = {name:content}
        self.current_action = 'have conversation with ' + ', '.join(chat_dict.keys())
        print("{} just {} at time {}".format(self.name, self.current_action, get_time_phase(self.global_time)))
        new_location = self.change_location(chat_dict = chat_dict)
        self.a_mem.location = new_location

    def execute_action(self, locations):
        for name in self.talk_dict.keys(): # target people accept talk
            try:
                self.talk_dict[name] = self.peoples_nearby[name].talk_dict.get(self.name, False)
            except:
                self.talk_dict[name]  = False

        if any(self.talk_dict.values()): # talk action
            chat_dict = self.talk_action()
            self.current_action = 'have conversation with ' + ', '.join(chat_dict.keys())
            print("{} just {} at time {}".format(self.name, self.current_action, get_time_phase(self.global_time)))
            new_location = self.change_location(chat_dict = chat_dict)
        else:        # no talk action
            prompt_path = "prompt/action.txt"
            prompt_input = []
            self.current_state = self.get_current_state()
            prompt_input.append(self.current_state)
            prompt_input.append(self.nearby_people)
            query = self.current_state + '. Now you are taking action for this time at {}'.format(self.location)
            relevant_memory, memory_state = self.a_mem.extract(query)
            prompt_input.append('{} : {}'.format(self.location, locations.get(self.location).description))
            prompt_input.append(self.plans)
            
            prompt_no_mem = prompt_generate(prompt_input, prompt_path)
            prompt = prompt_no_mem.replace(f"!<INPUT mem>!", relevant_memory)

            llm_output, token  = generate(self.prompt_meta.format(prompt))
            json_output = parse_json_from_content(llm_output)
            self.current_action = json_output.get('content') if json_output.get('content') else llm_output           
            if json_output.get('importance_score'):
                # Convert the matched importance score to an integer and check its range
                importance_score = int(json_output.get('importance_score'))
                if importance_score < 0 or importance_score > 9:
                    importance_score = 5 # Set default value if out of range
            else:
                importance_score = 5 # Set default value if no match was found    
            
            self.a_mem.add_event(self.global_day, self.global_time, self.current_action, importance_score, self.location, self.location)
            self.a_mem.query_with_mem.append({'query': query, 'memory_state': memory_state, 'action':llm_output, 'action_type':'event', 'prompt':self.prompt_meta.format(prompt_no_mem), 'base_memory':relevant_memory})
            if self.peoples_nearby:
                for name, agent in self.peoples_nearby.items():
                    ob_importance_score = agent.decide_observation_score(self.name, self.current_action)
                    agent.a_mem.add_observation(self.global_day, self.global_time, self.current_action, ob_importance_score, self.name, self.location)
            print("{} just {} in {} at time {}".format(self.name, self.current_action, self.location, get_time_phase(self.global_time)))
            new_location = self.change_location(action = self.current_action)
        self.a_mem.location = new_location
            
    def decide_observation_score(self, name, observation):
        prompt = 'Evaluate the significance of your observation about {} performing the action "{}" when recalling this memory in the future. Rate its importance on a scale from 0 (least important) to 9 (most important). Example Output: {{"importance score": 5}}'.format(name, observation)

        llm_output, token  = generate(self.prompt_meta.format(prompt))                
        importance_score_pattern = re.compile(r'"importance[_\s]?score"\s*:\s*(\d+)', re.IGNORECASE)
        importance_score_match = importance_score_pattern.search(llm_output)
        if importance_score_match:
            # Convert the matched importance score to an integer and check its range
            importance_score = int(importance_score_match.group(1))
            if importance_score < 0 or importance_score > 9:
                importance_score = 5 # Set default value if out of range
        else:
            importance_score = 5 # Set default value if no match was found       
        return importance_score
    
    def decide_to_talk(self, peoples_nearby):
        ''' input target agent name, description...
            decide wheather to talk.
        '''
        peoples_name = [name for name in self.peoples_nearby.keys()]
        peoples_description = ''.join(['Name: {}. {}'.format(agent.name, agent.brief_description) for name, agent in self.peoples_nearby.items()])
        prompt_path = "prompt/decide_to_talk.txt"
        prompt_input = []
        self.current_state = self.get_current_state()
        prompt_input.append(self.current_state)
        prompt_input.append(peoples_description)
        prompt_input.append(str(peoples_name))
        query = 'You are deciding whether to talk with {} during the {} at {}'.format(peoples_name, get_pure_time_phase(self.global_time), self.location)
        relevant_memory, memory_state = self.a_mem.extract(query)
        prompt_no_mem = prompt_generate(prompt_input, prompt_path)
        prompt = prompt_no_mem.replace(f"!<INPUT mem>!", relevant_memory)
        answer, token = generate(self.prompt_meta.format(prompt))
        decide_to_talk = []
        for name in peoples_name:
            if name.lower() in answer.lower():
                decide_to_talk.append(name)
        self.a_mem.query_with_mem.append({'query': query, 'memory_state': memory_state, 'action':answer, 'action_type':'decide_to_talk', 'prompt':self.prompt_meta.format(prompt_no_mem), 'base_memory':relevant_memory})

        return decide_to_talk
    
    def generative_conversation(self, agent, chat_history, n_talk):
        ''' input target agent name, description, and chat_history...
            decide talk content.
        '''
        prompt_path = "prompt/generative_conversation.txt"
        prompt_input = []
        self.current_state = self.get_current_state()
        prompt_input.append(self.current_state)
        prompt_input.append('Name: {}. '.format(agent.name) + agent.brief_description)
        if chat_history:
            last_message = str(chat_history[-1])
        else:
            last_message = ""

        query = self.current_state + '. Now you are talking with {} during this time at {}. The last thing they said was: "{}"'.format(agent.name, self.location, last_message)
        relevant_memory, memory_state = self.a_mem.extract(query)
        prompt_input.append(chat_history_to_str(chat_history))
        prompt_no_mem = prompt_generate(prompt_input, prompt_path)
        prompt = prompt_no_mem.replace(f"!<INPUT mem>!", relevant_memory)
        llm_output, token = generate(self.prompt_meta.format(prompt))
        json_output = parse_chat(llm_output)
        response = json_output.get('response') if json_output.get('response') else llm_output   
        n_talk = json_output.get('continue_the_conversation') if json_output.get('continue_the_conversation') else False   
        print('{}:{}'.format(self.name, response))
        self.a_mem.query_with_mem.append({'query': query, 'memory_state': memory_state, 'action':response, 'action_type':'talk', 'prompt':self.prompt_meta.format(prompt_no_mem), 'base_memory':relevant_memory})

        return response, n_talk
    
    def talk_action(self):
        for name, n_talk in self.talk_dict.items():
            chat_history = []
            while n_talk: # 修改prompt，确保合适的时候结束对话，防止对话无意义绕圈
                chat_response, n_talk = self.generative_conversation(self.peoples_nearby[name], chat_history, n_talk)
                chat_history.append([self.name, chat_response])
                if n_talk:
                    chat_response, n_talk = self.peoples_nearby[name].generative_conversation(self, chat_history, n_talk)
                    chat_history.append([name, chat_response])
            self.talk_dict[name] = False
            self.peoples_nearby[name].talk_dict[self.name] = False
            if chat_history:
                str_chat = chat_history_to_str(chat_history)
                chat_dict = {name:str_chat} 
                prompt = 'Assess the significance of your chat histoy as the memories when recalled in the future. Rate its importance on a scale from 0 (least important) to 9 (most important). chat history:{}. Example Output: {{"importance score": 4}}'.format(chat_history_to_str)
                llm_output, token  = generate(self.prompt_meta.format(prompt))                
                importance_score_pattern = re.compile(r'"importance[_\s]?score"\s*:\s*(\d+)', re.IGNORECASE)
                importance_score_match = importance_score_pattern.search(llm_output)
                if importance_score_match:
                    # Convert the matched importance score to an integer and check its range
                    importance_score = int(importance_score_match.group(1))
                    if importance_score < 0 or importance_score > 9:
                        importance_score = 5 # Set default value if out of range
                else:
                    importance_score = 5 # Set default value if no match was found                  
                self.a_mem.add_chat(self.global_day, self.global_time, str_chat, importance_score, name, self.location)
                self.peoples_nearby[name].a_mem.add_chat(self.global_day, self.global_time, str_chat, importance_score, self.name, self.location)
        return chat_dict    
    
    def change_location(self, **kwargs):
        if 'chat_dict' in kwargs:  
            last_action = ''
            for name, history in kwargs['chat_dict'].items():
                last_action += 'Talk with {} and chat history was: {}'.format(name, history)
        elif 'action' in kwargs:  # 如果关键字参数中有'action'
            last_action = kwargs['action']
        else:
            print("{} decided to stay at the current location {} at time {}".format(self.name, self.location, self.global_time + 1))
            return
        next_time_phase = get_time_phase(self.global_time + 1)
        self.working_state = next_time_phase.endswith("it's working time")
        work_phase = "must working in your workplace: {}, and can only change location for non workplace when you are in emrgency".format(self.workplace) if self.working_state else "you can do anything or go anywhere you like"
        locations_description = ''
        locations_name = []
        for name, location in self.all_locations.items():
            locations_description += '{} : {}'.format(name, location.description)
            locations_name.append(name)   
        self.current_state = self.get_current_state()
        query = self.current_state + '. Now you are in the location of {} and you need to decide whether to move to a new location at {}.'.format(self.location, next_time_phase)
        relevant_memory, memory_state = self.a_mem.extract(query)        
        prompt_no_mem = ('One hour ago, your state was "{}", and you spent the last hour {} \n Now, after one hours action, the next hour is "{}", In this time you {}. You have following memories:. Based on the descriptions of all locations: {}\n You must decide wheather and where to change your location for the next hour. Choose from the following location names: {}').format(self.current_state, last_action, next_time_phase, work_phase, locations_description, ', '.join(locations_name))
        prompt = ('One hour ago, your state was "{}", and you spent the last hour {} \n Now, after one hours action, the next hour is "{}", In this time you {}. You have following memories:{}. Based on the descriptions of all locations: {}\n You must decide wheather and where to change your location for the next hour. Choose from the following location names: {}').format(self.current_state, last_action, next_time_phase, work_phase, relevant_memory, locations_description, ', '.join(locations_name))
        decide, token  = generate(self.prompt_meta.format(prompt))
        self.a_mem.query_with_mem.append({'query': query, 'memory_state': memory_state, 'action':decide, 'action_type':'decide_location', 'prompt':self.prompt_meta.format(prompt_no_mem), 'base_memory':relevant_memory})
        change_index =  find_location_in_response(decide, locations_name)
        if change_index and locations_name[change_index] != self.location:
            self.location = locations_name[change_index]
            # f_work = 'for work' if self.working_state else ''
            print("{} moved to new location: {} at time {}".format(self.name, self.location, get_time_phase(self.global_time+1)))
            return locations_name[change_index]
        else:
            print("{} decided to stay at the current location {} at time {}".format(self.name, self.location, get_time_phase(self.global_time+1)))   
            return self.location   
        
        
        
         

        