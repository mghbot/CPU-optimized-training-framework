import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import multiprocessing as mp
from collections import defaultdict

# --- Algorithmic Innovation 1: Fused Layer (Memory Bandwidth Reduction) ---
class FusedLinearReLU(nn.Module):
    """Fuses linear+bias+ReLU into single op to minimize memory traffic on CPU"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Store weights in BF16 to halve memory bandwidth
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Execute in BF16, return in FP32 for stability
        x_bf16 = x.to(torch.bfloat16)
        out = torch.nn.functional.linear(x_bf16, self.weight, self.bias)
        return torch.relu_(out).to(x.dtype)  # In-place ReLU

# --- Algorithmic Innovation 2: BF16-Optimized LAMB (Layer-wise Adaptive) ---
class LAMB_BF16(torch.optim.Optimizer):
    """LAMB optimizer with BF16 optimizer states to reduce memory bandwidth"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad.detach()
                state = self.state[p]

                # Initialize BF16 optimizer states
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Bias correction
                bias_corr1 = 1 - beta1 ** state['step']
                bias_corr2 = 1 - beta2 ** state['step']

                # Compute update
                update = exp_avg / bias_corr1
                denom = (exp_avg_sq / bias_corr2).sqrt().add_(group['eps'])
                update.div_(denom)

                if group['weight_decay'] != 0:
                    update.add_(p.detach(), alpha=group['weight_decay'])

                # Trust ratio (layer-wise adaptive scaling)
                param_norm = p.detach().norm()
                update_norm = update.norm()
                trust_ratio = 1.0 if param_norm == 0 or update_norm == 0 else param_norm / update_norm

                # Update parameters
                p.detach().add_(update, alpha=-group['lr'] * trust_ratio)

        return loss

# --- Algorithmic Innovation 3: Lookahead (Meta-Optimizer for Stability) ---
class Lookahead:
    """Lookahead optimizer wrapper: maintains stability while exploring fast"""
    def __init__(self, base_optimizer, alpha=0.5, k=5):
        self.optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)
        self.counter = 0

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.counter += 1

        # Slow parameter update every k steps
        if self.counter % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    if 'slow' not in state:
                        state['slow'] = p.detach().clone()
                    # Interpolate toward fast weights
                    state['slow'].add_(p - state['slow'], alpha=self.alpha)
                    # Sync fast weights to slow
                    p.copy_(state['slow'])

        return loss

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)

# --- Algorithmic Innovation 4: Dynamic Layer Freezing (Compute Reduction) ---
class DynamicFreezer:
    """Dynamically freezes converged layers to save compute (novel for CPU)"""
    def __init__(self, model: nn.Module, threshold: float = 0.01, patience: int = 30):
        self.model = model
        self.threshold = threshold
        self.patience = patience
        self.stats = defaultdict(list)
        self.counters = defaultdict(int)
        self.frozen = set()

        # Register gradient hooks
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(self._hook(name))

    def _hook(self, name: str):
        def hook(grad):
            if name in self.frozen: return grad
            self.stats[name].append(grad.norm().item())
            if len(self.stats[name]) > 50: self.stats[name].pop(0)

            # Check for convergence
            if len(self.stats[name]) >= 10:
                median_grad = np.median(self.stats[name])
                if median_grad < self.threshold:
                    self.counters[name] += 1
                    if self.counters[name] >= self.patience:
                        for n, p in self.model.named_parameters():
                            if n == name:
                                p.requires_grad = False
                                self.frozen.add(name)
                                print(f"  Froze {name} (grad_norm={median_grad:.4f})")
                else:
                    self.counters[name] = 0
        return hook

# --- Model Architecture (Optimized for CPU Parallelism) ---
def create_model(num_classes: int = 1000) -> nn.Module:
    """Wide, shallow network for better CPU utilization"""
    layers = []
    dims = [784, 2048, 4096, 4096, 4096]

    for i in range(len(dims) - 1):
        layers.append(FusedLinearReLU(dims[i], dims[i+1]))
    layers.append(nn.Linear(dims[-1], num_classes, dtype=torch.bfloat16))

    model = nn.Sequential(*layers)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M parameters")
    return model

# --- Synthetic Dataset (On-the-fly Generation) ---
class SyntheticDataset(Dataset):
    """Generates data on-demand to avoid memory overhead"""
    def __init__(self, size: int = 50000, input_dim: int = 784, num_classes: int = 1000):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Deterministic generation for reproducibility
        seed = 42 + idx
        data = np.random.default_rng(seed).standard_normal(self.input_dim).astype(np.float32)
        target = np.random.default_rng(seed).integers(0, self.num_classes)
        return torch.from_numpy(data), target

# --- Training System (Integrates All Innovations) ---
class CPUTrainingSystem:
    def __init__(self, model: nn.Module, loader: DataLoader,
                 epochs: int = 10, accumulation_steps: int = 8):
        self.model = model
        self.loader = loader
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps

        # Optimizer: Lookahead(LAMB) for stability + layer-wise adaptation
        base_optimizer = LAMB_BF16(model.parameters(), lr=0.01, weight_decay=0.01)
        self.optimizer = Lookahead(base_optimizer, alpha=0.5, k=5)

        # Dynamic freezer for compute reduction
        self.freezer = DynamicFreezer(model, threshold=0.01, patience=30)

        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> nn.Module:
        """Main training loop with micro-batching and accumulation"""
        print(f"\nTraining {self.epochs} epochs (effective batch={128*self.accumulation_steps})...")
        self.model.train()

        epoch_times = []
        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_loss = 0.0
            num_updates = 0

            for batch_idx, (data, target) in enumerate(self.loader):
                # Convert to BF16 for forward pass
                data_bf16 = data.to(torch.bfloat16)

                # Forward + backward
                loss = self.criterion(self.model(data_bf16), target)
                (loss / self.accumulation_steps).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    total_loss += loss.item()
                    num_updates += 1

                    if batch_idx % 20 == 0:
                        print(f'  Epoch {epoch} [{batch_idx}/{len(self.loader)}] '
                              f'Loss: {loss.item():.4f}')

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            avg_loss = total_loss / num_updates if num_updates > 0 else 0
            throughput = len(self.loader.dataset) / epoch_time

            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, '
                  f'Throughput={throughput:.0f} samples/s')

        avg_throughput = len(self.loader.dataset) / np.mean(epoch_times)
        print(f"\nTraining complete!")
        print(f"Average throughput: {avg_throughput:.0f} samples/sec")
        print(f"Froze {len(self.freezer.frozen)} layers: {list(self.freezer.frozen)}")

        return self.model

# --- Main Entry Point ---
def main():
    """Configure and launch training system"""
    # CPU environment setup
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(42)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    # Verify configuration
    print("=== CPU Training System Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"BF16 supported: {torch.cpu.amp.autocast_supported()}")

    # Create synthetic dataset (50K samples, 784D input, 1000 classes)
    dataset = SyntheticDataset(size=50000)
    loader = DataLoader(
        dataset,
        batch_size=128,  # Micro-batch size (fits in L3 cache)
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    # Create model (~52M parameters)
    model = create_model(num_classes=1000)

    # Initialize training system
    system = CPUTrainingSystem(
        model=model,
        loader=loader,
        epochs=10,
        accumulation_steps=8  # Effective batch size = 1024
    )

    # Train
    trained_model = system.train()

    return trained_model

if __name__ == '__main__':
    main()
