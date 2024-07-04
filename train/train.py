import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CrossAttentionModel
from dataset import load_data
import json
import sys
import os 
from tqdm import tqdm
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_generation import generate_score





def id_to_data(raw_data_path, agent_id_list):
    with open(raw_data_path, 'r') as f:
        loaded_data = json.load(f)
    raw_data = [entry for entry in loaded_data if entry['id'] in agent_id_list]
    return raw_data

def logit_loss(output_score):
    loss = -torch.log(output_score + 1)
    return loss

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return time.strftime("%H:%M:%S", time.gmtime(time_dif))


class CustomScoreLoss(nn.Module):
    def __init__(self):
        super(CustomScoreLoss, self).__init__()

    def forward(self, output_scores):
        losses = -torch.log(output_scores + 1 + 1e-8)  # Add a small constant for numerical stability
        non_negative_losses = torch.max(losses, torch.tensor(0.0))
        return non_negative_losses.mean()


def train_model(model, save_path, data_loader, optimizer, criterion, device, num_top_memories, emb_path, raw_data_path, num_epochs, interval_num=50, resume=False):
    start_epoch = 0
    if resume and os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path))
        start_epoch = resume-1# 你需要一个方法来存储或确定从哪个epoch恢复
        print(f"Model loaded from {save_path}, resuming training from epoch {start_epoch+1}")
    total_batch = 0
    total_token = 0
    best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        total_samples = 0
        loop = tqdm(data_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]')

        for queries, memory_values, agent_id in data_loader:
            optimizer.zero_grad()
            queries = queries.to(device)  # 移动数据到 GPU
            memory_values = memory_values.to(device)  # 移动数据到 GPU
            
            agent_id_list = agent_id.tolist()
            raw_data = id_to_data(raw_data_path, agent_id_list)
            _, _, predicted_top_memory, predicted_top_indices = model(queries, memory_values, num_top_memories)
            l_t_i = predicted_top_indices.tolist()
            # use mem to llm
            loss_scores = [] # load top memories, real action, and prompt without memory
            for i in range(len(l_t_i)):
                indice = l_t_i[i]
                mem_datas = raw_data[i]['detail_mem'] # all memory nodes of agent in that action moment
                top_mem_text = [mem_datas[j] if j < len(mem_datas) else '' for j in indice]
                pre_memory = " ".join(top_mem_text)
                base_memory = raw_data[i]['base_memory']
                score, token = generate_score(raw_data[i]['prompt'], pre_memory, base_memory) # token assume, unit is k.
                total_token += token
                loss_scores.append(score)
            # Calculate loss, requires_grad=True
            loss = criterion(torch.tensor(loss_scores, dtype=torch.float32, requires_grad=True))
            loss.backward()
            optimizer.step()
            # Update total loss for average calculation later
            total_loss += loss.item() * len(queries)
            total_samples += len(queries)
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}] Processing...')
            loop.update(1)  # Ensure progress bar updates every iteration
            
            if total_batch % interval_num == 0: # early stop
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
            time_dif = get_time_dif(start_time)  # 定义 get_time_dif 来计算时间差
            msg = f'Iter: {total_batch:>6}, Train Loss: {loss.item():>5.2}, total_token: {total_token:.1f}k, Time: {time_dif} {improve}'
            loop.set_postfix_str(msg)

            total_batch += 1
            if total_batch - last_improve > 100:  # 1000是一个示例，你可以根据需要设置
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        average_loss = total_loss / total_samples
        print(f'End of Epoch {epoch+1}, Average Loss: {average_loss:.4f}')

                


def main():
    embed_size = 1536
    num_top_memories = 5
    batch_size = 16
    num_epochs=5
    raw_data_path = 'train\\dataset\\processed_data.json'
    emb_path = 'train\\dataset\\all_embeddings.json'
    save_path = 'train\\save_model\\mem_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentionModel(embed_size).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loader = DataLoader(load_data(raw_data_path, emb_path, max_mem_len=128), batch_size=batch_size, shuffle=True)
    criterion = CustomScoreLoss()
    train_model(model, save_path, data_loader, optimizer, criterion, device, num_top_memories, emb_path, raw_data_path, num_epochs=num_epochs, interval_num=50, resume=2)


if __name__ == "__main__":
    main()