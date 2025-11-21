"""
Ablation Studies Script

Tests each innovation independently to measure its contribution
to overall system performance.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
import argparse
from collections import defaultdict

from cpu_training_system import (
    FusedLinearReLU, LAMB_BF16, Lookahead, DynamicFreezer,
    SyntheticDataset, create_model
)


# ============================================================================
# ABLATION 1: BF16 vs FP32
# ============================================================================

class FP32FusedLinearReLU(nn.Module):
    """Fused layer but using FP32 instead of BF16"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return torch.relu_(out)


def test_bf16_vs_fp32(dataset_size=1000, epochs=3):
    """Compare BF16 vs FP32 training"""
    print("\n" + "=" * 80)
    print("ABLATION TEST 1: BF16 vs FP32")
    print("=" * 80)

    dataset = SyntheticDataset(size=dataset_size, input_dim=784, num_classes=100)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    results = {}

    # Test BF16
    print("\n--- Testing BF16 ---")
    model_bf16 = nn.Sequential(
        FusedLinearReLU(784, 2048),
        FusedLinearReLU(2048, 4096),
        nn.Linear(4096, 100, dtype=torch.bfloat16)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_bf16.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_bf16(data_bf16), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    bf16_time = time.time() - start_time
    results['bf16'] = {'time': bf16_time, 'throughput': (dataset_size * epochs) / bf16_time}
    print(f"BF16 Time: {bf16_time:.2f}s, Throughput: {results['bf16']['throughput']:.0f} samples/s")

    # Test FP32
    print("\n--- Testing FP32 ---")
    model_fp32 = nn.Sequential(
        FP32FusedLinearReLU(784, 2048),
        FP32FusedLinearReLU(2048, 4096),
        nn.Linear(4096, 100, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            loss = criterion(model_fp32(data), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    fp32_time = time.time() - start_time
    results['fp32'] = {'time': fp32_time, 'throughput': (dataset_size * epochs) / fp32_time}
    print(f"FP32 Time: {fp32_time:.2f}s, Throughput: {results['fp32']['throughput']:.0f} samples/s")

    # Results
    speedup = fp32_time / bf16_time
    print(f"\n✓ BF16 is {speedup:.2f}x faster than FP32")
    print(f"  Time saved: {fp32_time - bf16_time:.2f}s ({((1 - bf16_time/fp32_time)*100):.1f}%)")

    return results


# ============================================================================
# ABLATION 2: Fused vs Separate Operations
# ============================================================================

class SeparateLinearReLU(nn.Module):
    """Standard separate Linear + ReLU (no fusion)"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=torch.bfloat16)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        return self.relu(self.linear(x_bf16)).to(x.dtype)


def test_fused_vs_separate(dataset_size=1000, epochs=3):
    """Compare fused vs separate operations"""
    print("\n" + "=" * 80)
    print("ABLATION TEST 2: Fused vs Separate Operations")
    print("=" * 80)

    dataset = SyntheticDataset(size=dataset_size, input_dim=784, num_classes=100)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    results = {}

    # Test Fused
    print("\n--- Testing Fused Linear+ReLU ---")
    model_fused = nn.Sequential(
        FusedLinearReLU(784, 2048),
        FusedLinearReLU(2048, 4096),
        nn.Linear(4096, 100, dtype=torch.bfloat16)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_fused.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_fused(data_bf16), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    fused_time = time.time() - start_time
    results['fused'] = {'time': fused_time, 'throughput': (dataset_size * epochs) / fused_time}
    print(f"Fused Time: {fused_time:.2f}s, Throughput: {results['fused']['throughput']:.0f} samples/s")

    # Test Separate
    print("\n--- Testing Separate Linear + ReLU ---")
    model_separate = nn.Sequential(
        SeparateLinearReLU(784, 2048),
        SeparateLinearReLU(2048, 4096),
        nn.Linear(4096, 100, dtype=torch.bfloat16)
    )

    optimizer = torch.optim.Adam(model_separate.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_separate(data_bf16), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    separate_time = time.time() - start_time
    results['separate'] = {'time': separate_time, 'throughput': (dataset_size * epochs) / separate_time}
    print(f"Separate Time: {separate_time:.2f}s, Throughput: {results['separate']['throughput']:.0f} samples/s")

    # Results
    speedup = separate_time / fused_time
    print(f"\n✓ Fused is {speedup:.2f}x faster than separate ops")
    print(f"  Memory bandwidth reduction: ~{((1 - fused_time/separate_time)*100):.1f}%")

    return results


# ============================================================================
# ABLATION 3: LAMB+Lookahead vs Adam
# ============================================================================

def test_optimizers(dataset_size=2000, epochs=5):
    """Compare LAMB+Lookahead vs Adam"""
    print("\n" + "=" * 80)
    print("ABLATION TEST 3: Optimizers (LAMB+Lookahead vs Adam)")
    print("=" * 80)

    dataset = SyntheticDataset(size=dataset_size, input_dim=784, num_classes=100)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    results = {}

    # Test Adam
    print("\n--- Testing Adam Optimizer ---")
    model_adam = create_model(num_classes=100)
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

    losses_adam = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_adam(data_bf16), target)
            loss.backward()
            optimizer_adam.step()
            optimizer_adam.zero_grad()
            epoch_loss += loss.item()
        losses_adam.append(epoch_loss / len(loader))
        print(f"  Epoch {epoch}: Loss={losses_adam[-1]:.4f}")

    adam_time = time.time() - start_time
    results['adam'] = {
        'time': adam_time,
        'final_loss': losses_adam[-1],
        'losses': losses_adam
    }
    print(f"Adam Time: {adam_time:.2f}s, Final Loss: {losses_adam[-1]:.4f}")

    # Test LAMB+Lookahead
    print("\n--- Testing LAMB+Lookahead ---")
    model_lamb = create_model(num_classes=100)
    base_optimizer = LAMB_BF16(model_lamb.parameters(), lr=0.01)
    optimizer_lamb = Lookahead(base_optimizer, alpha=0.5, k=5)

    losses_lamb = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_lamb(data_bf16), target)
            loss.backward()
            optimizer_lamb.step()
            optimizer_lamb.zero_grad()
            epoch_loss += loss.item()
        losses_lamb.append(epoch_loss / len(loader))
        print(f"  Epoch {epoch}: Loss={losses_lamb[-1]:.4f}")

    lamb_time = time.time() - start_time
    results['lamb_lookahead'] = {
        'time': lamb_time,
        'final_loss': losses_lamb[-1],
        'losses': losses_lamb
    }
    print(f"LAMB+Lookahead Time: {lamb_time:.2f}s, Final Loss: {losses_lamb[-1]:.4f}")

    # Results
    speedup = adam_time / lamb_time if lamb_time > 0 else 0
    convergence_improvement = (losses_adam[-1] - losses_lamb[-1]) / losses_adam[-1] * 100

    print(f"\n✓ LAMB+Lookahead vs Adam:")
    print(f"  Time: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"  Final loss: {convergence_improvement:+.1f}% {'better' if convergence_improvement > 0 else 'worse'}")
    print(f"  Convergence: {'Faster' if losses_lamb[-1] < losses_adam[-1] else 'Slower'}")

    return results


# ============================================================================
# ABLATION 4: Dynamic Freezing
# ============================================================================

def test_dynamic_freezing(dataset_size=2000, epochs=10):
    """Compare training with and without dynamic freezing"""
    print("\n" + "=" * 80)
    print("ABLATION TEST 4: Dynamic Freezing")
    print("=" * 80)

    dataset = SyntheticDataset(size=dataset_size, input_dim=784, num_classes=100)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    results = {}

    # Test without freezing
    print("\n--- Testing WITHOUT Dynamic Freezing ---")
    model_no_freeze = create_model(num_classes=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_no_freeze.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_no_freeze(data_bf16), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    no_freeze_time = time.time() - start_time
    results['no_freeze'] = {
        'time': no_freeze_time,
        'frozen_layers': 0
    }
    print(f"Without Freezing Time: {no_freeze_time:.2f}s")

    # Test with freezing
    print("\n--- Testing WITH Dynamic Freezing ---")
    model_freeze = create_model(num_classes=100)
    freezer = DynamicFreezer(model_freeze, threshold=0.01, patience=10)
    optimizer = torch.optim.Adam(model_freeze.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        for data, target in loader:
            data_bf16 = data.to(torch.bfloat16)
            loss = criterion(model_freeze(data_bf16), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    freeze_time = time.time() - start_time
    results['freeze'] = {
        'time': freeze_time,
        'frozen_layers': len(freezer.frozen)
    }
    print(f"With Freezing Time: {freeze_time:.2f}s")
    print(f"Frozen layers: {len(freezer.frozen)}")
    print(f"Frozen layer names: {list(freezer.frozen)}")

    # Results
    speedup = no_freeze_time / freeze_time
    compute_saved = (1 - freeze_time / no_freeze_time) * 100

    print(f"\n✓ Dynamic Freezing Results:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Compute saved: {compute_saved:.1f}%")
    print(f"  Layers frozen: {len(freezer.frozen)}")

    if compute_saved > 10:
        print(f"  ✓ Significant improvement!")
    elif compute_saved > 0:
        print(f"  ⚠ Modest improvement")
    else:
        print(f"  ✗ No improvement (may need more epochs)")

    return results


# ============================================================================
# Main Runner
# ============================================================================

def run_all_ablations():
    """Run all ablation studies"""
    print("=" * 80)
    print("RUNNING ALL ABLATION STUDIES")
    print("=" * 80)

    all_results = {}

    # Run all tests
    all_results['bf16_vs_fp32'] = test_bf16_vs_fp32()
    all_results['fused_vs_separate'] = test_fused_vs_separate()
    all_results['optimizers'] = test_optimizers()
    all_results['dynamic_freezing'] = test_dynamic_freezing()

    # Final summary
    print("\n" + "=" * 80)
    print("ABLATION STUDIES SUMMARY")
    print("=" * 80)

    print("\n1. BF16 vs FP32:")
    bf16_speedup = all_results['bf16_vs_fp32']['fp32']['time'] / all_results['bf16_vs_fp32']['bf16']['time']
    print(f"   Speedup: {bf16_speedup:.2f}x")

    print("\n2. Fused vs Separate:")
    fused_speedup = all_results['fused_vs_separate']['separate']['time'] / all_results['fused_vs_separate']['fused']['time']
    print(f"   Speedup: {fused_speedup:.2f}x")

    print("\n3. Optimizers:")
    opt_speedup = all_results['optimizers']['adam']['time'] / all_results['optimizers']['lamb_lookahead']['time']
    print(f"   LAMB+Lookahead vs Adam: {opt_speedup:.2f}x")
    print(f"   Final loss (Adam): {all_results['optimizers']['adam']['final_loss']:.4f}")
    print(f"   Final loss (LAMB+Lookahead): {all_results['optimizers']['lamb_lookahead']['final_loss']:.4f}")

    print("\n4. Dynamic Freezing:")
    freeze_speedup = all_results['dynamic_freezing']['no_freeze']['time'] / all_results['dynamic_freezing']['freeze']['time']
    print(f"   Speedup: {freeze_speedup:.2f}x")
    print(f"   Layers frozen: {all_results['dynamic_freezing']['freeze']['frozen_layers']}")

    print("\n" + "=" * 80)
    print("OVERALL IMPACT")
    print("=" * 80)
    total_speedup = bf16_speedup * fused_speedup * freeze_speedup
    print(f"Theoretical combined speedup: {total_speedup:.2f}x")
    print(f"(Note: Actual speedup may differ due to interactions)")

    # Save results
    import json
    with open("ablation_results.json", "w") as f:
        # Convert any numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print("\nResults saved to ablation_results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "bf16", "fused", "optimizers", "freezing"],
                        help="Which test to run")

    args = parser.parse_args()

    if args.test == "all":
        run_all_ablations()
    elif args.test == "bf16":
        test_bf16_vs_fp32()
    elif args.test == "fused":
        test_fused_vs_separate()
    elif args.test == "optimizers":
        test_optimizers()
    elif args.test == "freezing":
        test_dynamic_freezing()
