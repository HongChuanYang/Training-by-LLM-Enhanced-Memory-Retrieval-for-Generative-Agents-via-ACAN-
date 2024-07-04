# Training-by-LLM-Enhanced-Memory-Retrieval-for-Generative-Agents-via-ACAN-
This repository contains the code for the paper "Training by LLM: Enhanced Memory Retrieval for Generative Agents via an Auxiliary Cross Attention Network". Our work introduces a novel approach to memory retrieval for generative agents, leveraging the power of large language models (LLMs) to enhance the effectiveness and precision of memory recall in complex agent-based simulations.


# Introduction
This project builds upon the concept of generative agents in a multi-agent, multi-location simulated environment. We utilize an Auxiliary Cross Attention Network (ACAN) to improve memory retrieval by calculating attention weights between an agent's current state and a memory repository, thereby retrieving the most relevant memories for the agent's current context. This method significantly surpasses traditional memory retrieval techniques by integrating LLMs to assess and guide the training of our model.

# Usage

To run the simulation, follow these steps:

1. Configure character and location information in the `game_simulation` folder.

2. Set the API key in `utils/text_generation.py`.

3. Use `run.py` to set up basic simulation information. Set `memory_type = 'base'` to run the simulation using the base memory retrieval method to generate base simulation data.

4. Generate training data using the notebook `train/agent_data_preprocess.ipynb`.

5. Train the model using `train/train.py`.

6. Change the `memory_type` in `run.py` to `'acan'` to run the simulation with the ACAN memory retrieval mode.

