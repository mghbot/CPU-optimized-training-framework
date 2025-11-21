"""
Baseline Comparison Script

Compares the CPU-optimized training system against standard PyTorch training
to measure the speedup and effectiveness of the innovations.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from cpu_training_system import SyntheticDataset, create_model as create_optimized_model, CPUTrainingSystem


class StandardModel(nn.Module):
    """Standard PyTorch model (FP32, no fusions)"""
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        dims = [784, 2048, 4096, 4096, 4096]

        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(dims[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StandardTrainingSystem:
    """Standard PyTorch training (Adam, FP32, no tricks)"""
    def __init__(self, model: nn.Module, loader: DataLoader,
                 epochs: int = 10, accumulation_steps: int = 8):
        self.model = model
        self.loader = loader
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps

        # Standard Adam optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> nn.Module:
        """Standard training loop"""
        print(f"\nTraining {self.epochs} epochs (standard PyTorch)...")
        self.model.train()

        epoch_times = []
        epoch_losses = []

        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_loss = 0.0
            num_updates = 0

            for batch_idx, (data, target) in enumerate(self.loader):
                # Forward + backward (FP32)
                loss = self.criterion(self.model(data), target)
                (loss / self.accumulation_steps).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_loss += loss.item()
                    num_updates += 1

                    if batch_idx % 20 == 0:
                        print(f'  Epoch {epoch} [{batch_idx}/{len(self.loader)}] '
                              f'Loss: {loss.item():.4f}')

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            avg_loss = total_loss / num_updates if num_updates > 0 else 0
            epoch_losses.append(avg_loss)
            throughput = len(self.loader.dataset) / epoch_time

            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, '
                  f'Throughput={throughput:.0f} samples/s')

        avg_throughput = len(self.loader.dataset) / np.mean(epoch_times)
        print(f"\nTraining complete!")
        print(f"Average throughput: {avg_throughput:.0f} samples/sec")

        return self.model, epoch_times, epoch_losses


def measure_memory_usage(func):
    """Measure peak memory usage during function execution"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024**3  # GB

    result = func()

    peak_memory = process.memory_info().rss / 1024**3  # GB
    memory_used = peak_memory - initial_memory

    return result, memory_used


def run_comparison(dataset_size=5000, epochs=5, num_workers=12):
    """Run side-by-side comparison"""

    print("=" * 80)
    print("BASELINE COMPARISON: Optimized vs Standard PyTorch")
    print("=" * 80)

    # Setup
    torch.manual_seed(42)
    dataset = SyntheticDataset(size=dataset_size, input_dim=784, num_classes=1000)

    print(f"\nDataset size: {dataset_size}")
    print(f"Epochs: {epochs}")
    print(f"Num workers: {num_workers}")

    # ========== STANDARD PYTORCH ==========
    print("\n" + "=" * 80)
    print("1. STANDARD PYTORCH TRAINING (Baseline)")
    print("=" * 80)

    loader_standard = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model_standard = StandardModel(num_classes=1000)
    system_standard = StandardTrainingSystem(
        model_standard, loader_standard, epochs=epochs, accumulation_steps=8
    )

    start_time = time.time()
    _, standard_times, standard_losses = system_standard.train()
    total_time_standard = time.time() - start_time

    # ========== OPTIMIZED SYSTEM ==========
    print("\n" + "=" * 80)
    print("2. OPTIMIZED CPU TRAINING SYSTEM")
    print("=" * 80)

    loader_optimized = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model_optimized = create_optimized_model(num_classes=1000)
    system_optimized = CPUTrainingSystem(
        model_optimized, loader_optimized, epochs=epochs, accumulation_steps=8
    )

    start_time = time.time()
    _ = system_optimized.train()
    total_time_optimized = time.time() - start_time

    # Extract metrics (need to modify CPUTrainingSystem to return these)
    # For now, we'll estimate from prints

    # ========== RESULTS ==========
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    avg_time_standard = np.mean(standard_times)
    avg_time_optimized = total_time_optimized / epochs

    throughput_standard = dataset_size / avg_time_standard
    throughput_optimized = dataset_size / avg_time_optimized

    speedup = avg_time_standard / avg_time_optimized
    throughput_improvement = throughput_optimized / throughput_standard

    print("\nThroughput:")
    print(f"  Standard PyTorch: {throughput_standard:.1f} samples/sec")
    print(f"  Optimized System: {throughput_optimized:.1f} samples/sec")
    print(f"  Improvement: {throughput_improvement:.2f}x")

    print("\nTime per Epoch:")
    print(f"  Standard PyTorch: {avg_time_standard:.2f}s")
    print(f"  Optimized System: {avg_time_optimized:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")

    print("\nTotal Training Time:")
    print(f"  Standard PyTorch: {total_time_standard:.2f}s")
    print(f"  Optimized System: {total_time_optimized:.2f}s")
    print(f"  Time Saved: {total_time_standard - total_time_optimized:.2f}s")

    print("\nFinal Loss:")
    print(f"  Standard PyTorch: {standard_losses[-1]:.4f}")
    print(f"  Optimized System: (check training logs)")

    print("\nOptimized System Features:")
    print(f"  - Frozen layers: {len(system_optimized.freezer.frozen)}")
    print(f"  - Uses BF16: Yes")
    print(f"  - Uses fused layers: Yes")
    print(f"  - Uses LAMB+Lookahead: Yes")
    print(f"  - Uses dynamic freezing: Yes")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if speedup > 1.2:
        print(f"✓ OPTIMIZED SYSTEM IS {speedup:.2f}x FASTER")
    elif speedup > 1.0:
        print(f"⚠ OPTIMIZED SYSTEM IS SLIGHTLY FASTER ({speedup:.2f}x)")
    else:
        print(f"✗ OPTIMIZED SYSTEM IS SLOWER ({speedup:.2f}x)")

    print(f"\nThe optimized system achieved:")
    print(f"  - {throughput_improvement:.1%} higher throughput")
    print(f"  - {((1 - avg_time_optimized/avg_time_standard) * 100):.1f}% reduction in training time")

    if system_optimized.freezer.frozen:
        print(f"  - {len(system_optimized.freezer.frozen)} layers frozen dynamically")

    return {
        "standard": {
            "throughput": throughput_standard,
            "time_per_epoch": avg_time_standard,
            "total_time": total_time_standard,
            "final_loss": standard_losses[-1]
        },
        "optimized": {
            "throughput": throughput_optimized,
            "time_per_epoch": avg_time_optimized,
            "total_time": total_time_optimized,
            "frozen_layers": len(system_optimized.freezer.frozen)
        },
        "speedup": speedup,
        "improvement": throughput_improvement
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument("--dataset-size", type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of data loader workers")

    args = parser.parse_args()

    results = run_comparison(
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        num_workers=args.num_workers
    )

    # Save results
    import json
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to baseline_results.json")
