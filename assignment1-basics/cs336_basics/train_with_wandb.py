from transformer import *
from optimizer import *
from utils import *
from bpe import *
from tokenizer import *
import glob
import swanlab # 导入swanlab库
import pickle
import numpy as np
import torch
import os


def train():
    # 1. Hyperparameters and configurations are grouped together
    config = {
        'd_model': 32,
        'num_layers': 2,
        'num_heads': 2,
        'd_ff': 128,
        'rope_theta': 10000,
        'lr': 3e-3,
        'epochs': 3,
        'dataset_path': '../data/tiny.txt',
        'batch_size': 2,
        'context_length': 20,
        'device': 'cpu',
        'vocab_path': './vocab.pkl',
        'merges_path': './merges.pkl',
    }

    # 初始化 swanlab run
    # 项目名称可以随意更改
    swanlab.init(project="train-test-gpu", config=config)

    # 2. Function calls use unpacked dictionary for cleaner code
    train_model(config)


def train_model(config):
    # Unpack configurations for easy access
    d_model = config['d_model']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    rope_theta = config['rope_theta']
    lr = config['lr']
    epochs = config['epochs']
    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    context_length = config['context_length']
    device = config['device']
    vocab_path = config['vocab_path']
    merges_path = config['merges_path']

    # 3. Data loading section
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    tokenized_file_path = "tokenized_data.bin"

    try:
        with open(tokenized_file_path, "rb") as f:
            pass
        print(f"Using existing tokenized data file: {tokenized_file_path}")
    except FileNotFoundError:
        print("Tokenizing and saving dataset to disk...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = f.read()
        dataset = tokenizer.encode(data)
        with open(tokenized_file_path, "wb") as f:
            f.write(np.array(dataset, dtype=np.uint16).tobytes())
            print(f"Tokenized data saved to {tokenized_file_path}")

    print(f"Loading dataset from {tokenized_file_path} using np.memmap...")
    memmap_dataset = np.memmap(tokenized_file_path, dtype=np.uint16, mode='r')
    total_tokens = len(memmap_dataset)
    print(f"Total tokens loaded: {total_tokens}")

    split_index = int(total_tokens * 0.9)
    train_dataset = memmap_dataset[:split_index]
    val_dataset = memmap_dataset[split_index:]
    print(f"Training tokens: {len(train_dataset)}, Validation tokens: {len(val_dataset)}")

    # 4. Model and optimizer initialization
    model = Transformer(
        vocab_size=len(vocab),
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device
    )
    opt = AdamW(model.parameters(), lr=lr)

    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        i_step = 0
        while True:
            x, y = get_batch(train_dataset, batch_size, context_length, device)
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            loss = cross_entropy(y_pred, y)

            opt.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), 1e-2)
            opt.step()

            if i_step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {i_step}, Training Loss: {loss.cpu().item():.4f}")
                # 使用 swanlab.log 记录训练损失，并指定一个步数
                swanlab.log({"train/loss": loss.cpu().item()}, step=epoch * (len(train_dataset) // batch_size) + i_step)
            i_step += 1
            if i_step * batch_size > len(train_dataset):
                break

        # 6. Evaluation loop
        model.eval()
        val_loss_sum = 0
        num_val_batches = 0
        while True:
            x_val, y_val = get_batch(val_dataset, batch_size, context_length, device)
            with torch.no_grad():
                y_val_pred = model(x_val)
                y_val_pred = y_val_pred.view(-1, y_val_pred.shape[-1])
                y_val = y_val.view(-1)
                val_loss = cross_entropy(y_val_pred, y_val)
                val_loss_sum += val_loss.cpu().item()
            num_val_batches += 1
            if num_val_batches * batch_size > len(val_dataset):
                break

        avg_val_loss = val_loss_sum / num_val_batches
        print(f"--------------------------------------------------")
        print(f"Epoch {epoch} finished. Average Validation Loss: {avg_val_loss:.4f}")
        print(f"--------------------------------------------------")

        # 使用 swanlab.log 记录验证损失
        swanlab.log({"val/loss": avg_val_loss, "epoch": epoch})

        save_checkpoint(model, opt, epoch, f"checkpoint_epoch{epoch}.pt")

    # 结束 swanlab run
    swanlab.finish()


def find_latest_checkpoint(checkpoint_dir='.'):
    """
    在指定目录下找到最新的检查点文件。

    Args:
        checkpoint_dir (str): 检查点文件所在的目录。

    Returns:
        str: 最新的检查点文件路径，如果没有找到则返回 None。
    """
    # 查找所有符合 'checkpoint_epoch*.pt' 模式的文件
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*.pt'))

    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def test(prompt="hello", max_output_token_num=10):
    # 1. Hyperparameters and configurations are grouped together
    config = {
        'd_model': 32,
        'num_layers': 2,
        'num_heads': 2,
        'd_ff': 128,
        'rope_theta': 10000,
        'lr': 3e-3,
        'epochs': 3,
        'dataset_path': '../data/tiny.txt',
        'batch_size': 2,
        'context_length': 20,
        'device': 'cpu',
        'vocab_path': './vocab.pkl',
        'merges_path': './merges.pkl',
    }

    # 2. Function calls use unpacked dictionary for cleaner code
    test_model(config, prompt, max_output_token_num)


def test_model(config, prompt, max_output_token_num):
    # Unpack configurations for easy access
    d_model = config['d_model']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    rope_theta = config['rope_theta']
    lr = config['lr']
    context_length = config['context_length']
    device = config['device']
    vocab_path = config['vocab_path']
    merges_path = config['merges_path']

    # 3. Data loading section
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])


    model = Transformer(
        vocab_size=len(vocab),
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device
    )
    opt = AdamW(model.parameters(), lr=lr)

    latest_checkpoint_path = find_latest_checkpoint()
    load_checkpoint(latest_checkpoint_path, model, opt)

    # 5. Training loop
    model.eval()

    x = tokenizer.encode(prompt)
    # print(x)
    x = torch.tensor(x, dtype=torch.long)
    x = x.unsqueeze(0)
    # print(x)

    prompt_token_num = x.shape[1]
    res = []
    # token_num = prompt_token_num
    while True:
        with torch.no_grad():
            y = model(x)
            # print(x.shape, y.shape)
            _, indices = y[0, y.shape[1] - 1].max(dim=-1)
            next_token_id = indices.item()
            next_token = tokenizer.decode([next_token_id])[0]
            if next_token == '<|endoftext|>':
                break
            # print(next_token)
            res.append(next_token)

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            x = torch.cat((x, next_token_tensor), dim=1)
            if x.shape[1] > context_length:
                x = x[:, 1:]
            if (y.shape[1] - prompt_token_num) >= (max_output_token_num):
                break
    print('output is:', res)

if __name__ == '__main__':
    # train_2()
    # test()
    train()