import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory.base_memory import AssociativeMemory
from train.model import CrossAttentionModel


class AcanMemory(AssociativeMemory):
    def extract(self, query):
    # Custom logic for the enhanced extract method
        top_k = 5
        max_mem_len = 128
        query_embedding = self.get_embedding(query)
        queries = torch.tensor(query_embedding, dtype=torch.float32) 
        mem_list = []
        memory_embeddings = []
        for node in self.id_to_node.values():
            mem_list.append(node)
            memory_embedding = self.embeddings[node.description]
            memory_embeddings.append(torch.tensor(memory_embedding, dtype=torch.float32))
        if memory_embeddings:
            # 使用第一个元素的形状来创建零向量进行填充
            padding = [torch.zeros_like(memory_embeddings[0]) for _ in range(max_mem_len - len(memory_embeddings))]
            memory_embeddings.extend(padding)
        else:
            # 如果没有记忆，创建一个全零的填充张量列表
            padding_shape = query_embedding.shape
            # padding_shape = self.embeddings[item['query']].shape
            memory_embeddings = [torch.zeros(padding_shape, dtype=torch.float32) for _ in range(self.max_mem_len)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memory_state = []
        for key in self.id_to_node.keys():
            memory_state.append(key)
        predicted_top_indices = self.load_model(device, queries, torch.stack(memory_embeddings), top_k)
        top_nodes = [mem_list[i] for i in predicted_top_indices]
        memory_summary = " ".join([f"{node.type} memory about {node.object} : {node.description} in {node.location} at day:{node.created_day}{node.created};\n " for node in top_nodes])
        return memory_summary, memory_state

    def load_model(self, device, queries, memory_values, top_k):
        embed_size = 1536
        save_path = 'train\\save_model\\mem_model.pth'
        model = CrossAttentionModel(embed_size).to(device) 
        predicted_top_indices = self.test_model(model, save_path, queries, memory_values, top_k, device)
        return predicted_top_indices
        
    def test_model(self, model, save_path, queries, memory_values, top_k, device):
    # def test_model(self, model, data_loader, criterion, device, save_path, num_top_memories, raw_data_path):
        model.load_state_dict(torch.load(save_path))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad(): 
            queries = queries.to(device)
            memory_values = memory_values.to(device)
            _, _, predicted_top_memory, predicted_top_indices = model(queries.unsqueeze(0) , memory_values.unsqueeze(0) , top_k)
            l_t_i = predicted_top_indices.tolist()
        return l_t_i[0]