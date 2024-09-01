from openai import OpenAI
import json
import subprocess
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'      # AutoDL
#os.environ['HF_HOME'] = 'D:/tmp/cache'     # win11
#os.environ["HF_HOME"] = "/mnt/d/tmp/cache"  # wsl2
import time
from tigerscore import TIGERScorer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from torch.optim import Adam, AdamW
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def query(input):
    message = [
        {"role": "system", "content": ""},
        {"role": "user", "content": input},
    ]
    hypo_output = None
    received = False
    while (not received) or (hypo_output is None):
        try:
            hypo_output = client.chat.completions.create(
                model=chatbot,
                messages=message,
                stream=False,
                max_tokens=960
            )
            received = True
        except:
            time.sleep(1)
    return hypo_output.choices[0].message.content

def query_concurrently(prompts, instruction, input_context):
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(query, p + instruction + input_context) for p in prompts]
        results = [future.result() for future in futures]
    return results

def query_concurrently_sample_size(sample_size, instruction, input_context):
    with ThreadPoolExecutor(max_workers=sample_size) as executor:
        futures = [executor.submit(query, instruction + input_context) for _ in range(sample_size)]
        results = [future.result() for future in futures]
    return results

def tigerloss(instruction, input_context, output):
    received = False
    while not received:
        try:
            result = scorer.score([instruction], [output], [input_context])
            received = True
        except:
            time.sleep(1)
    score = result[0].get('score', 0)  # 使用 get 方法提供默认值 0
    loss = - score if score is not None else 0
    return loss

def pmi():
    result = []
    with open(pmi_data, 'r', encoding='utf-8') as f:
        for line in f:
            result.append(line.strip('\n'))
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(unique)
    return ngram_index_list

def train(epochs, sample_size):

    best_eval_result = float('inf')
    best_alphas = None
    best_epoch = 0
    eval_results = []
    test_results = []
    train_losses = []
    prev_alphas = alphas.clone()
    patience = 0


    for epoch in range(epochs):
        epoch_loss = 0
        random.shuffle(train_data)
        for item in train_data:
            prompts_probs = F.softmax(alphas)
            prompts_dist = torch.distributions.Categorical(prompts_probs)
            loss_list = []
            prompts_discrete_indices_list = []
            prompts = []
            for k in range(sample_size):
                prompts_discrete_indices = prompts_dist.sample()
                prompts_discrete_indices_list.append(prompts_discrete_indices)
                prompts_discrete_ngram_list = []
                indices_list = prompts_discrete_indices.int().tolist()
                for idx in indices_list:
                    prompts_discrete_ngram_list.append(ngram_list[idx])
                prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                prompts.append(prompts_discrete)
            hypo_outputs = query_concurrently(prompts, item['instruction'], item['input_context'])

            for hypo_output in hypo_outputs:
                loss = tigerloss(item['instruction'], item['input_context'], hypo_output)
                loss_list.append(loss)

            loss_avg = sum(loss_list) / sample_size
            epoch_loss += loss_avg
            print(loss_list)
            prompt_optimizer.zero_grad()
            # 计算log(P)的梯度
            derivative = torch.zeros_like(alphas)
            for prompts_discrete_indices, loss in zip(prompts_discrete_indices_list, loss_list):
                log_probs = prompts_probs.gather(1, prompts_discrete_indices.unsqueeze(1)).squeeze()
                log_prob = torch.log(log_probs).sum()
                log_prob.backward(retain_graph=True)
                derivative += alphas.grad * (loss - loss_avg)
            alphas.grad = derivative / (sample_size-1)
            print("总alphas.grad:", torch.linalg.norm(alphas.grad))
            torch.nn.utils.clip_grad_norm_(alphas, 5)
            prompt_optimizer.step()
            print("epoch:", epoch, "loss_avg:", loss_avg)

        epoch_loss /= len(train_data)
        train_losses.append(epoch_loss)
        print("Epoch:", epoch, "Training Loss:", epoch_loss)

        # 计算alphas参数的变化
        alphas_change = torch.norm(alphas - prev_alphas)
        prev_alphas = alphas.detach().clone()  # 更新prev_alphas为当前的alphas

        eval_result = evaluate()
        eval_results.append(eval_result)

        # 检查是否有性能提升或alphas参数变化是否小于阈值
        if eval_result < best_eval_result:
            best_eval_result = eval_result
            best_alphas = alphas.detach().clone()
            best_epoch = epoch
            print(f"New best eval result: {best_eval_result} at epoch {epoch} with alphas change: {alphas_change}")
            patience = 0
        else:
            patience += 1
            print(f"Patience counter: {patience}")

        if alphas_change < alpha_change_threshold or patience >= patience_threshold:
            print(f"Stopping training. Alphas change: {alphas_change}, Patience: {patience}")
            break

        # 保存检查点
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"data/checkpoint/checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'alphas': alphas,
                'optimizer': prompt_optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")


        # 如果cuda内存不够，清理一下
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    save_path = "data/best_alphas.pt"
    torch.save(best_alphas, save_path)
    print("Finished training")

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_results, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    return

def evaluate(sample_size=5):
    eval_result = 0
    for item in test_data:
        prompts_probs = F.softmax(alphas)
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        prompts = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            prompts.append(prompts_discrete)
        hypo_outputs = query_concurrently(prompts, item['instruction'], item['input_context'])

        for hypo_output in hypo_outputs:
            loss = tigerloss(item['instruction'], item['input_context'], hypo_output)
            loss_list.append(loss)
        loss_avg = sum(loss_list) / sample_size
        print("eval_loss_avg:", loss_avg)
        eval_result += loss_avg

    eval_result /= len(test_data)
    return eval_result

def test(sample_size=5):
    test_result = []
    origin_result = []
    for item in test_data:
        prompts_probs = F.softmax(alphas)
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        origin_loss_list = []
        prompts = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            prompts.append(prompts_discrete)
        outputs = query_concurrently(prompts, item['instruction'], item['input_context'])
        outputs_origin = query_concurrently_sample_size(sample_size, item['instruction'], item['input_context'])
        for output, output_origin in zip(outputs, outputs_origin):
            loss = tigerloss(item['instruction'], item['input_context'], output)
            origin_loss = tigerloss(item['instruction'], item['input_context'], output_origin)
            loss_list.append(loss)
            origin_loss_list.append(origin_loss)

        loss_avg = sum(loss_list) / sample_size
        origin_loss_avg = sum(origin_loss_list) / sample_size
        test_result.append(loss_avg)
        origin_result.append(origin_loss_avg)
        print("test_loss_avg:", loss_avg)
        print("origin_loss_avg:", origin_loss_avg)
    test_result /= len(test_data)
    origin_result /= len(test_data)
    print("Finished testing")
    return test_result, origin_result

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    alphas.load_state_dict(checkpoint['alphas'])
    prompt_optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss}")
    return start_epoch, loss

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # set up scorer
    #scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B")  # on GPU
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU, about 5 instances per seconds
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU

    client = OpenAI(api_key="sk-0c2e4c0ec7444bc7924a645788c4dd24", base_url="https://api.deepseek.com/v1")
    chatbot = "deepseek-chat"
    # client = OpenAI(api_key="sk-HPOmC99SEkbTxygFd28Nba6785yOocrSpDqzLu94FafdXqOW", base_url="https://api.moonshot.cn/v1")
    # chatbot = "moonshot-v1-8k"
    # 读取数据
    data = json.load(open("data/cut_data.json", 'r', encoding='utf-8'))
    pmi_data = "data/vocab.txt"
    train_data = data[:int(0.7 * len(data))]
    test_data = data[int(0.7 * len(data)):]

    # 设置超参数
    learning_rate = 1e-3
    ngram_list = pmi()
    prompt_length = 35
    prompt_search_space = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    alphas = torch.FloatTensor([[5] * prompt_search_space] * prompt_length)
    alphas.requires_grad = True
    temperature = 0.5
    alpha_change_threshold = 1e-2
    patience_threshold = 5
    checkpoint_interval = 5

    # 设置优化器
    prompt_optimizer = AdamW([{
        "params": [alphas],
        "weight_decay": 0.1,
    }], lr=learning_rate)


    train(epochs=20, sample_size=10)
    test_result, origin_result = test(5)

    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")