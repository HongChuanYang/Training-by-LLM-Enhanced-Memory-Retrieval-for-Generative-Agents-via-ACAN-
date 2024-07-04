import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class MemoryDataset(Dataset):
    def __init__(self, data, embeddings, max_mem_len):
        self.data = data
        self.embeddings = embeddings  # 嵌入字典或加载的模型
        self.max_mem_len = max_mem_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query_embedding = torch.tensor(self.embeddings[item['query']], dtype=torch.float32) 
        memory_embeddings = [torch.tensor(self.embeddings[memory], dtype=torch.float32) for memory in item['memory']] # tensor
        # 如果 memory_embeddings 不为空，进行填充
        if memory_embeddings:
            # 使用第一个元素的形状来创建零向量进行填充
            padding = [torch.zeros_like(memory_embeddings[0]) for _ in range(self.max_mem_len - len(memory_embeddings))]
            memory_embeddings.extend(padding)
        else:
            # 如果没有记忆，创建一个全零的填充张量列表
            padding_shape = query_embedding.shape
            # padding_shape = self.embeddings[item['query']].shape
            memory_embeddings = [torch.zeros(padding_shape, dtype=torch.float32) for _ in range(self.max_mem_len)]

        return query_embedding, torch.stack(memory_embeddings), item['id']


# 数据加载函数
def load_data(data_path, emb_path, max_mem_len):
    with open(data_path, 'r') as f:
        loaded_data = json.load(f)
    with open(emb_path, 'r') as f:
        loaded_emb = json.load(f)
    return MemoryDataset(loaded_data, loaded_emb, max_mem_len)
