import json
from game_simulation.locations import Location
from agent.agent import Agent
import os
from utils.utils import get_pure_time_phase, reverse_get_pure_time_phase

locations_file_path = "game_simulation/locations.json"
avatars_file_path = "game_simulation/avatars.json"


# read map information to networkx as a cycle
with open(locations_file_path, 'r') as f:
    town_areas = json.load(f)
with open(avatars_file_path, 'r') as f:
    town_people = json.load(f)
 

# time setting
global_time = 0
times_per_day = 16
repeat_day = 2

memory_type = 'acan' # 'base' or 'acan', acan must training first

# build agent and map
locations = {}
agents = {}
for name, description in town_areas.items():
    locations[name] = Location(name, description)
for name, details in town_people.items():
    agents[name] = Agent(name, details, locations, memory_type)

last_memory = list(agents.values())[-1].a_mem.get_last_memory('event')
if last_memory:
    global_time = reverse_get_pure_time_phase(last_memory.created) + 1
    global_day = last_memory.created_day - 1 
else:
    global_time = 0
    global_day = 0
total_day = global_day+repeat_day
repeats = 1
# start stimulation  
while global_day < total_day:  
    for time in range(global_time, times_per_day):
        #log_output for one repeat
        log_output = ""
        pure_time = get_pure_time_phase(time)
        print(f"============================ REPEAT {repeats} ============================")
        print("====================== day:{}, time :{} ======================\n".format(global_day + 1, pure_time))
        for agent_name, agent in agents.items():
            # agent.a_mem.clear() 
            agent.plan(agents, time, global_day)
        # Execute action
        for agent_name, agent in agents.items():
            action = agent.execute_action(locations)            
        for agent_name, agent in agents.items():
            agent.a_mem.decay()
            agent.a_mem.save() 
        repeats += 1
    global_time = 0
    global_day += 1
