import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import random
import sys
import time as Time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from Modules_ori import *  # MultiHeadAttention, PositionwiseFeedForward

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DreamRec with Consistency Model Training.")
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='Dataset: yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int,
                        default=100, help='Random seed')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size')
    parser.add_argument('--layers', type=int, default=1, help='GRU layers')
    parser.add_argument('--hidden_factor', type=int,
                        default=64, help='Embedding size')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='Noise levels for consistency training')
    parser.add_argument('--beta_start', type=float,
                        default=0.0001, help='Beta start for schedule')
    parser.add_argument('--beta_end', type=float,
                        default=0.02, help='Beta end for schedule')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--l2_decay', type=float,
                        default=0.0, help='L2 regularization')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--dropout_rate', type=float,
                        default=0.1, help='Dropout rate')
    parser.add_argument('--p', type=float, default=0.1,
                        help='Masking probability')
    parser.add_argument('--report_epoch', action='store_true',
                        help='Report each epoch')
    parser.add_argument(
        '--beta_sche', choices=['linear', 'exp', 'cosine'], default='exp', help='Beta schedule')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='Type of diffuser MLP: mlp1 or mlp2')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer: adam, adamw, adagrad, rmsprop')
    parser.add_argument('--save_model_dir', type=str,
                        default='ckpt', help='Path to save model checkpoint')
    parser.add_argument('--load_model_num', type=int,
                        default=0, help='load model checkpoint')
    parser.add_argument('--eval', action='store_true',
                        default=False, help='evaluate the model')
    return parser.parse_args()


args = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    return 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))


class DiffusionSchedule:
    """
    Provides q_sample and one-step sampling for consistency models.
    """

    def __init__(self, timesteps, beta_start, beta_end, beta_sche='exp'):
        self.timesteps = timesteps
        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            self.betas = exp_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        b_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return a_t * x_start + b_t * noise

    def sample(self, model, h, device):
        # one-step sampling: start from pure Gaussian noise at highest level
        x = torch.randn_like(h).to(device)
        t = torch.full((h.shape[0],), self.timesteps-1,
                       device=device, dtype=torch.long)
        return model(x, h, t)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device
        # embeddings
        self.item_embeddings = nn.Embedding(item_num+1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        # transformer
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout)
        self.ff = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        # step embedding
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size)
        )
        # diffuser
        if diffuser_type == 'mlp1':
            self.diffuser = nn.Linear(hidden_size*3, hidden_size)
        else:
            self.diffuser = nn.Sequential(
                nn.Linear(hidden_size*3, hidden_size*2),
                nn.GELU(),
                nn.Linear(hidden_size*2, hidden_size)
            )

    def forward(self, x, h, step):
        t = self.step_mlp(step)
        return self.diffuser(torch.cat((x, h, t), dim=1))

    def cacu_x(self, x):
        return self.item_embeddings(x)

    def cacu_h(self, states, len_states, p):
        emb = self.item_embeddings(states)
        emb = emb + \
            self.positional_embeddings(torch.arange(
                self.state_size, device=self.device))
        seq = self.emb_dropout(emb)
        mask = (states != self.item_num).float().unsqueeze(-1).to(self.device)
        seq = seq * mask
        x = self.ln1(seq)
        x = self.mh_attn(x, seq)
        x = self.ff(self.ln2(x)) * mask
        x = self.ln3(x)
        last = extract_axis_1(x, len_states-1)
        h = last.squeeze()
        # dropout in hidden
        drop = (torch.rand(h.size(0), device=self.device)
                > p).float().unsqueeze(1)
        h = h*drop + self.none_embedding(torch.zeros(
            1, dtype=torch.long, device=self.device)).expand_as(h)*(1-drop)
        return h

    def predict(self, states, len_states, diff):
        h = self.cacu_h(states, len_states, p=0.0)
        # one-step sampling
        x0 = diff.sample(self, h, self.device)
        emb = self.item_embeddings.weight  # (item_num+1, hidden_size)
        scores = torch.matmul(x0, emb.T)
        return scores


def evaluate(model, test_data, diff, device):
    df = pd.read_pickle(os.path.join(data_directory, test_data))
    batch_size = 100
    hit = [0]*3
    ndcg = [0]*3
    total = 0
    topk = [10, 20, 50]
    seqs, lens, tgt = list(df['seq'].values), list(
        df['len_seq'].values), list(df['next'].values)
    for i in range(len(seqs)//batch_size):
        bs = torch.LongTensor(
            np.array(seqs[i*batch_size:(i+1)*batch_size], dtype=np.int64)).to(device)
        bl = torch.tensor(lens[i*batch_size:(i+1)*batch_size],
                          dtype=torch.long, device=device)
        bt = torch.LongTensor(
            np.array(tgt[i*batch_size:(i+1)*batch_size], dtype=np.int64)).to(device)
        # states = torch.LongTensor(np.array(bs, dtype=np.int64)).to(device)
        scores = model.predict(bs, bl, diff)
        _, top_idx = scores.topk(100, dim=1)
        arr = top_idx.cpu().numpy()
        sorted_arr = np.flip(arr, axis=1).copy()
        calculate_hit(sorted_arr, topk, bt, hit, ndcg)
        total += batch_size
    print(f"HR@10 NDCG@10 HR@20 NDCG@20 HR@50 NDCG@50")
    metrics = []
    for idx in range(len(topk)):
        metrics.append(hit[idx]/total)
        metrics.append(ndcg[idx]/total)
    metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in metrics]
    print(" ".join(f"{m:.4f}" for m in metrics))
    return hit[1]/total  # HR@20


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    data_directory = './data/'+args.data
    stats = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    seq_size = stats['seq_size'][0]
    item_num = stats['item_num'][0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Tenc(args.hidden_factor, item_num, seq_size,
                 args.dropout_rate, args.diffuser_type, device)
    diff = DiffusionSchedule(
        args.timesteps, args.beta_start, args.beta_end, args.beta_sche)
    model.to(device)

    if args.load_model_num:
        path = os.path.join(args.save_model_dir, f'ckpt_{args.data}_{args.load_model_num}.pt')
        model.load_state_dict(torch.load(
            path, map_location=device))
        print(f"Loaded model from {path}")
    
    if args.eval:
        test_start = Time.time()
        print('--- Test ---')
        _ = evaluate(model, 'test_data.df', diff, device)
        print("Test Cost: " + Time.strftime("%H: %M: %S",
            Time.gmtime(Time.time()-test_start)))
        print('------------------')
        sys.exit(0)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)

    train_df = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    num_batches = int(train_df.shape[0]/args.batch_size)

    for epoch in range(args.load_model_num, args.load_model_num + args.epoch):
        # t0=Time.time()
        pbar = tqdm(range(num_batches),
                    desc=f'Epoch {epoch+1}/{args.load_model_num + args.epoch}', unit='batch')
        batch_loss = 0
        N = 0
        for _ in pbar:
            batch = train_df.sample(n=args.batch_size).to_dict()
            seq = torch.LongTensor(
                np.array(list(batch['seq'].values()), dtype=np.int64)).to(device)
            lens = torch.tensor(
                list(batch['len_seq'].values()), dtype=torch.long, device=device)
            tgt = torch.LongTensor(
                np.array(list(batch['next'].values()), dtype=np.int64)).to(device)

            x_start = model.cacu_x(tgt)
            h = model.cacu_h(seq, lens, args.p)

            # consistency training step
            noise = torch.randn_like(x_start)
            t = torch.randint(0, args.timesteps,
                              (args.batch_size,), device=device)
            delta = torch.randint(
                1, args.timesteps, (args.batch_size,), device=device)
            s = torch.clamp(t+delta, max=args.timesteps-1)
            x_t = diff.q_sample(x_start, t, noise)
            x_s = diff.q_sample(x_start, s, noise)
            pred_t = model(x_t, h, t)
            pred_s = model(x_s, h, s)

            # Full-range supervision: anchor both levels to x_start
            loss_sup_t = F.mse_loss(pred_t, x_start)
            loss_sup_s = F.mse_loss(pred_s, x_start)
            # Consistency loss
            loss_cons = F.mse_loss(pred_s, pred_t)
            loss = loss_sup_t + loss_sup_s + loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            N += len(batch)
            pbar.set_postfix(loss=batch_loss/N)

        # if args.report_epoch:
        #     print(f"Epoch {epoch+1:03d} Loss {loss.item():.4f} Time {Time.strftime('%H:%M:%S', Time.gmtime(Time.time()-t0))}")
        if (epoch + 1) % 10 == 0:
            os.makedirs(args.save_model_dir, exist_ok=True)
            path = os.path.join(args.save_model_dir, f'ckpt_{args.data}_{epoch + 1}.pt')
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")

            val_start = Time.time()
            print('--- Validation ---')
            _ = evaluate(model, 'val_data.df', diff, device)
            print("Validation Cost: " + Time.strftime("%H: %M: %S",
                  Time.gmtime(Time.time()-val_start)))
            test_start = Time.time()
            print('--- Test ---')
            _ = evaluate(model, 'test_data.df', diff, device)
            print("Test Cost: " + Time.strftime("%H: %M: %S",
                  Time.gmtime(Time.time()-test_start)))
            print('------------------')
