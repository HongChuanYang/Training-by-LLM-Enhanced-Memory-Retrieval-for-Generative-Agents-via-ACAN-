import json
from utils.utils import get_pure_time_phase
import os 
from utils.text_generation import generate_embedding
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ConceptNode: 
  def __init__(self, created_day, created, node_id, node_count, node_type, description, poignancy, object, location, recency): 
    self.node_id = node_id
    self.node_count = node_count
    # self.type_count = type_count
    self.type = node_type # thought / event / chat
    self.created = get_pure_time_phase(created) if isinstance(created, int) else created
    self.created_day = created_day
    self.object = object
    self.description = description
    self.poignancy = poignancy
    self.recency = recency
    self.location = location
    
  def calculate_score(self, memory_embedding, query_embedding, min_max_scalers):
    epsilon = 1e-10
    relevance = np.dot(memory_embedding, query_embedding) / (np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding) + epsilon)

    # 归一化
    poignancy_normalized = min_max_scalers['poignancy'].transform([[self.poignancy]])[0][0]
    recency_normalized = min_max_scalers['recency'].transform([[self.recency]])[0][0]
    relevance_normalized = min_max_scalers['relevance'].transform([[relevance]])[0][0]
    retrieval_score = poignancy_normalized + recency_normalized + relevance_normalized
    return retrieval_score
    
  
  @staticmethod
  def create_node_from_dict(node_id, node_dict):

    return ConceptNode(
        created_day=node_dict["created_day"],
        created=node_dict["created"],
        node_count=node_dict["node_count"],
        node_id=node_id,
        # type_count=node_dict["type_count"],
        node_type=node_dict["type"],
        description=node_dict["description"],
        poignancy=node_dict["poignancy"],
        object=node_dict["object"],
        location=node_dict["location"],
        recency = node_dict["recency"]
      )




class AssociativeMemory: 
  def __init__(self, agent_mem_path, location):     
    self.agent_mem_path = agent_mem_path
    self.location = location
    self.query_with_mem = []
    # self.clear() # only activate when need clear memory
    self.load()
    
  def load(self):
    nodes_file_path = os.path.join(self.agent_mem_path, "nodes.json")
    embeddings_file_path = os.path.join(self.agent_mem_path, "embeddings.json")
    location_file_path = os.path.join(self.agent_mem_path, "location.json")
    query_with_mem_file_path = os.path.join(self.agent_mem_path, "query_with_mem.json")
    if os.path.exists(nodes_file_path):
        with open(nodes_file_path, "r") as infile:
            nodes_data = json.load(infile)
            self.id_to_node = {key: ConceptNode.create_node_from_dict(key, value) for key, value in nodes_data.items()}
    else:
      self.id_to_node = {}
    self.location = json.load(open(location_file_path)) if os.path.exists(location_file_path) else self.location
    self.embeddings = json.load(open(embeddings_file_path)) if os.path.exists(embeddings_file_path) else {}
    self.query_with_mem = json.load(open(query_with_mem_file_path)) if os.path.exists(query_with_mem_file_path) else self.query_with_mem 
    
  def add_thought(self, created_day, created, description, poignancy, object, location):
    node_count = len(self.id_to_node.keys()) + 1
    # type_count = len(self.seq_thought) + 1
    node_type = "thought"
    node_id = f"node_{str(node_count)}"
    node = ConceptNode(created_day, created, node_id, node_count, node_type, description, poignancy, object, location, 1.0)
    # self.seq_thought[0:0] = [node]
    embedding = self.get_embedding(description)
    self.id_to_node[node_id] = node 
    return node
  
  def add_event(self, created_day, created, description, poignancy, object, location):
    node_count = len(self.id_to_node.keys()) + 1
    # type_count = len(self.seq_event) + 1
    node_type = "event"
    node_id = f"node_{str(node_count)}"
    node = ConceptNode(created_day, created, node_id, node_count, node_type, description, poignancy, object, location, 1.0)
    # self.seq_event[0:0] = [node]
    embedding = self.get_embedding(description)
    self.id_to_node[node_id] = node 
    return node
  
  def add_observation(self, created_day, created, description, poignancy, object, location):
    node_count = len(self.id_to_node.keys()) + 1
    # type_count = len(self.seq_event) + 1
    node_type = "observe"
    node_id = f"node_{str(node_count)}"
    node = ConceptNode(created_day, created, node_id, node_count, node_type, description, poignancy, object, location, 1.0)
    embedding = self.get_embedding(description)
    # self.seq_event[0:0] = [node]
    # self.embeddings[description] = embedding
    self.id_to_node[node_id] = node 
    return node
  
  def add_chat(self, created_day, created, description, poignancy, object, location):
    node_count = len(self.id_to_node.keys()) + 1
    # type_count = len(self.seq_chat) + 1
    node_type = "chat"
    node_id = f"node_{str(node_count)}"
    node = ConceptNode(created_day, created, node_id, node_count, node_type, description, poignancy, object, location, 1.0)
    # self.seq_chat[0:0] = [node]
    embedding = self.get_embedding(description)
    self.id_to_node[node_id] = node 
    return node
  
  def get_last_memory(self, mem_type):
    #根据输出的mem_type，找出node.type一致的，node_id最后的一个node
    last_memory = None
    for node in self.id_to_node.values():
        if node.type == mem_type:
            if last_memory is None or node.node_count > last_memory.node_count:
                last_memory = node
    return last_memory
  
  def node_to_mem(self, id_to_node):
    r = dict()
    for count in range(len(id_to_node.keys()), 0, -1): 
        node_id = f"node_{str(count)}"
        node = id_to_node[node_id]
        r[node_id] = dict()
        r[node_id]["node_count"] = node.node_count
        # r[node_id]["type_count"] = node.type_count
        r[node_id]["type"] = node.type
        r[node_id]["created_day"] = node.created_day
        r[node_id]["created"] = node.created
        r[node_id]["object"] = node.object
        r[node_id]["location"] = node.location
        r[node_id]["description"] = node.description
        r[node_id]["poignancy"] = node.poignancy
        r[node_id]["recency"] = node.recency
    return r
    
  
  def save(self):
    #test 
    # self.add_chat(1, 0, "str_chat", 5, "James", generate_embedding("str_chat")) # test
    mem_nodes = self.node_to_mem(self.id_to_node)
    # self.query_with_mem.append({'query': query, 'memory_state': self.id_to_node})
    os.makedirs(self.agent_mem_path, exist_ok=True)
    with open(os.path.join(self.agent_mem_path, "nodes.json"), "w") as outfile:
      json.dump(mem_nodes, outfile)
    with open(os.path.join(self.agent_mem_path, "embeddings.json"), "w") as outfile:
      json.dump(self.embeddings, outfile)
    with open(os.path.join(self.agent_mem_path, "location.json"), "w") as outfile:
      json.dump(self.location, outfile)
    with open(os.path.join(self.agent_mem_path, "query_with_mem.json"), "w") as outfile:
      json.dump(self.query_with_mem, outfile)

          
          
  def clear(self):
      nodes_file_path = os.path.join(self.agent_mem_path, "nodes.json")
      embeddings_file_path = os.path.join(self.agent_mem_path, "embeddings.json")
      location_file_path = os.path.join(self.agent_mem_path, "location.json")

      # 删除文件
      if os.path.exists(nodes_file_path):
          os.remove(nodes_file_path)
      if os.path.exists(embeddings_file_path):
          os.remove(embeddings_file_path)
      if os.path.exists(location_file_path):
          os.remove(location_file_path)
      # # 清空字典
      # self.id_to_node.clear()
      # self.embeddings.clear()
      
  def decay(self):
    # 遍历所有节点，并使得node.recency = node.recency*0.995
    for node in self.id_to_node.values():
      node.recency *= 0.995
      
  def calculate_min_max_scalers(self, query_embedding):
        epsilon = 1e-10  # 避免除以零的情况

        poignancies = []
        recencies = []
        relevances = []
        for node in self.id_to_node.values():
            memory_embedding = self.embeddings[node.description]
            relevance = np.dot(memory_embedding, query_embedding) / (np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding) + epsilon)
            poignancies.append(node.poignancy)
            recencies.append(node.recency)
            relevances.append(relevance)
            
        if not poignancies:
          return {
              'poignancy': MinMaxScaler().fit(np.array([[0], [1]])),
              'recency': MinMaxScaler().fit(np.array([[0], [1]])),
              'relevance': MinMaxScaler().fit(np.array([[0], [1]]))
          }

        # 使用 MinMaxScaler 进行归一化
        poignancy_scaler = MinMaxScaler()
        recency_scaler = MinMaxScaler()
        relevance_scaler = MinMaxScaler()

        poignancy_scaler.fit(np.array(poignancies).reshape(-1, 1))
        recency_scaler.fit(np.array(recencies).reshape(-1, 1))
        relevance_scaler.fit(np.array(relevances).reshape(-1, 1))

        return {
            'poignancy': poignancy_scaler,
            'recency': recency_scaler,
            'relevance': relevance_scaler
        }
      

      
  def extract(self, query):
    # 提取最近的10个节点
    top_k = 5
    query_embedding = self.get_embedding(query)
    # min_max = self.calculate_min_max(query_embedding)
    min_max_scalers = self.calculate_min_max_scalers(query_embedding)
    scores = []
    for node in self.id_to_node.values():
      memory_embedding = self.embeddings[node.description]
      retrieval_score = node.calculate_score(memory_embedding, query_embedding, min_max_scalers)
      scores.append((retrieval_score, node))
    top_nodes = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    # 合并为一条记忆总结
    memory_summary = " ".join([f"{node[1].type} memory about {node[1].object} : {node[1].description} in {node[1].location} at day:{node[1].created_day}{node[1].created};\n " for node in top_nodes])
    # 加一个方法，保存query和此时的记忆库。
    memory_state = []
    for key in self.id_to_node.keys():
      memory_state.append(key)
    return memory_summary, memory_state
  
  def get_embedding(self, text):
    try:
        # 尝试从字典中获取嵌入
        embedding = self.embeddings[text]
    except:
        # 如果抛出 KeyError 异常，说明字典中没有该键，生成一个新的嵌入
        embedding = generate_embedding(text)
        self.embeddings[text] = embedding  # 将新生成的嵌入保存到字典中
    return embedding