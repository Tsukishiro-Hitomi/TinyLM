import argparse
import csv
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

from tinylm import AdamW, Linear, RoPE, TransformerLM, cross_entropy, get_batch, learning_rate_schedule, save_checkpoint
from _cli_utils import config_defaults, config_path_from_argv, resolve_device

DEVICE = 'cpu'
CONTEXT = 256


def model(ablation='none', vocab_size=10000, tie_weights=False):
    rope = RoPE(theta=10000, d_k=32, max_seq_len=CONTEXT, device=DEVICE)
    net = TransformerLM(d_model=512, num_heads=16, d_ff=1344, rope=rope,
                        vocab_size=vocab_size, num_layers=4, device=DEVICE)
    for block in net.blocks:
        if ablation == 'no_norm':
            def forward(self, x):
                y = x + self.attention(x, self.rope)
                return y + self.swiglu(y)
            block.forward = types.MethodType(forward, block)
        elif ablation == 'post_norm':
            def forward(self, x):
                y = self.rmsnorm1(x + self.attention(x, self.rope))
                return self.rmsnorm2(y + self.swiglu(y))
            block.forward = types.MethodType(forward, block)
        elif ablation == 'no_rope':
            block.rope = None
        elif ablation == 'silu':
            class SiLUFFN(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.first = Linear(512, 2048)
                    self.second = Linear(2048, 512)
                def forward(self, x):
                    x = self.first(x)
                    return self.second(x * torch.sigmoid(x))
            block.swiglu = SiLUFFN()
    if ablation == 'no_norm':
        net.rmsnorm = torch.nn.Identity()
    if tie_weights:
        torch.nn.init.trunc_normal_(net.embedding.embedding, mean=0.0, std=0.02, a=-0.06, b=0.06)
        net.final_linear.W = net.embedding.embedding
    return net.to(DEVICE)


@torch.no_grad()
def validation(net, data, batches=32, batch_size=16):
    net.eval()
    # A fixed, repeatable set of samples for fair comparisons between runs.
    state = torch.random.get_rng_state()
    torch.manual_seed(336)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, CONTEXT, DEVICE)
        total += cross_entropy(net(x), y).item()
    torch.random.set_rng_state(state)
    return total / batches


def main():
    description = 'Train or evaluate the Assignment 1 Transformer language model.'
    config_path = config_path_from_argv(description)
    defaults = config_defaults(config_path, 'training')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', help='TOML file containing a [training] table.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a checkpoint instead of training.')
    parser.add_argument('--checkpoint', default='runs/20261028-080941/checkpoints/checkpoint40000')
    parser.add_argument('--lr', type=float, default=0.0003, help='Peak learning rate.')
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--batch', type=int, default=32, help='Training batch size.')
    parser.add_argument('--run-name', default=None, help='Output directory name below --runs-dir.')
    parser.add_argument('--ablation', choices=['none', 'no_norm', 'post_norm', 'no_rope', 'silu'], default='none')
    parser.add_argument('--dataset', choices=['tinystories', 'owt'], default='tinystories')
    parser.add_argument('--tie-weights', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=None, help='Stop after this wall-clock budget.')
    parser.add_argument('--device', default='auto', help='Torch device (default: auto; e.g. cpu, cuda, cuda:0, mps).')
    parser.add_argument('--data-dir', default='data', help='Directory containing <dataset>_{train,valid}.bin.')
    parser.add_argument('--runs-dir', default='runs', help='Directory for run outputs.')
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    global DEVICE
    DEVICE = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    train = np.memmap(data_dir / f'{args.dataset}_train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(data_dir / f'{args.dataset}_valid.bin', dtype=np.uint16, mode='r')
    torch.manual_seed(336)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(336)
    net = model(args.ablation, 32000 if args.dataset == 'owt' else 10000, args.tie_weights)
    if args.evaluate:
        ckpt = args.checkpoint
        step, state, _ = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        net.load_state_dict(state)
        print(f'checkpoint={ckpt} step={step} val_loss={validation(net, val):.6f} val_tokens={32*16*CONTEXT}', flush=True)
        return
    run = Path(args.runs_dir) / (args.run_name or f'lr_sweep_{args.lr:g}_seed336')
    run.mkdir(parents=True, exist_ok=False)
    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=0.1)
    start = time.perf_counter()
    with (run / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'wall_time_s', 'train_loss', 'val_loss', 'lr'])
        for step in range(1, args.steps + 1):
            lr = learning_rate_schedule(step, args.lr, args.lr * 0.1, max(1, int(args.steps * .05)), args.steps)
            for group in opt.param_groups:
                group['lr'] = lr
            net.train()
            x, y = get_batch(train, args.batch, CONTEXT, DEVICE)
            loss = cross_entropy(net(x), y)
            if not torch.isfinite(loss) or loss.item() > 100:
                print(f'DIVERGED step={step} loss={loss.item()}', flush=True)
                return
            opt.zero_grad()
            loss.backward()
            opt.step()
            time_limit = args.max_seconds is not None and time.perf_counter() - start >= args.max_seconds
            if step % max(1, min(500, args.steps // 20)) == 0 or step == args.steps or time_limit:
                value = validation(net, val, batches=8)
                writer.writerow([step, time.perf_counter()-start, loss.item(), value, lr])
                f.flush()
                print(f'step={step} train={loss.item():.5f} val={value:.5f} lr={lr:.6g}', flush=True)
                if not math.isfinite(value) or value > 100:
                    print('DIVERGED validation', flush=True)
                    return
            if time_limit:
                print(f'TIME_LIMIT step={step}', flush=True)
                break
        save_checkpoint(net, opt, step, run / 'checkpoint_final')


if __name__ == '__main__':
    main()
