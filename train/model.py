import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModel(nn.Module):
    def __init__(self, embed_size):
        super(CrossAttentionModel, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))  # 根号dk
        self.embed_size = embed_size


    def forward(self, query, memory_values, top_k=1):
        # Process the query to match the required shape [1, embed_size]
        query = self.query(query).unsqueeze(1)  # Query shape: [batch_size, 1, embed_size]
        
        memory_values = memory_values.transpose(0, 1)  # New shape: [memory_size, batch_size, embed_size]
        # Process keys and values
        key = self.key(memory_values)  # Key shape: [memory_size, batch_size, embed_size]
        value = self.value(memory_values)  # Value shape: [memory_size, batch_size, embed_size] 
        key = key.transpose(0, 1).transpose(1, 2)  # [batch_size, embed_size, memory_size]

        attention_scores = torch.bmm(query, key) / self.scale.to(query.device)  # # [batch_size, 1, memory_size] Scaling by sqrt(d_k)
        attention_probs = F.softmax(attention_scores, dim=-1)  # Normalize scores to probabilities
        value = value.transpose(0, 1)  # [batch_size, memory_size, embed_size]
        output = torch.bmm(attention_probs, value)  # [batch_size, 1, embed_size]
        output = output.squeeze(1)  # [batch_size, embed_size]， 代表着query注意力下，依据权重综合考虑了记忆库中记忆后，给出的综合表示
        top_values, top_indices = torch.topk(attention_probs.squeeze(1), k=top_k, dim=1)  # [batch_size, top_k]
        top_memories = memory_values.transpose(0, 1) #[batch_size, memory_size, embed_size]
        top_memories = torch.gather(top_memories, 1, top_indices.unsqueeze(-1).expand(-1, -1, self.embed_size))


        return output, attention_probs.squeeze(1), top_memories, top_indices
    