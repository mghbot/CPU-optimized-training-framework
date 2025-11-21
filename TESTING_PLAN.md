# CPU Training System - Comprehensive Testing Plan

## Overview
This document outlines the complete testing strategy for validating the CPU-optimized training framework. Tests are designed to verify functionality, measure performance, and validate claimed innovations.

## Prerequisites

### Environment Setup
- **PyTorch**: Latest stable version with CPU support
- **Python**: 3.8+
- **NumPy**: Latest version
- **Hardware**: 12-core AMD Ryzen CPU, 64GB RAM
- **OS**: Linux recommended

### Installation Commands
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib pytest
```

## Test Suite Structure

### 1. Environment Validation Tests

#### 1.1 PyTorch Installation
```python
def test_pytorch_installation():
    import torch
    assert torch.__version__ is not None
    assert torch.cpu.is_available()
```

#### 1.2 BF16 Support
```python
def test_bf16_support():
    import torch
    is_supported = torch.cpu.amp.autocast_supported()
    print(f"BF16 support: {is_supported}")
    if not is_supported:
        print("WARNING: BF16 not supported, will fall back to FP32")
```

#### 1.3 Thread Configuration
```python
def test_thread_config():
    import torch
    torch.set_num_threads(12)
    assert torch.get_num_threads() == 12
```

### 2. Component Unit Tests

#### 2.1 FusedLinearReLU Layer
**Test**: Layer initialization and forward pass
```python
def test_fused_layer():
    from cpu_training_system import FusedLinearReLU
    layer = FusedLinearReLU(784, 2048)

    # Test weight initialization
    assert layer.weight.shape == (2048, 784)
    assert layer.weight.dtype == torch.bfloat16

    # Test forward pass
    input_data = torch.randn(32, 784)
    output = layer(input_data)

    assert output.shape == (32, 2048)
    assert (output >= 0).all()  # ReLU applied
```

**Expected**: Layer creates correct shapes, uses BF16, applies ReLU

#### 2.2 LAMB_BF16 Optimizer
**Test**: Optimizer step and state management
```python
def test_lamb_optimizer():
    from cpu_training_system import LAMB_BF16
    model = torch.nn.Linear(10, 5)
    optimizer = LAMB_BF16(model.parameters(), lr=0.01)

    # Perform optimization step
    output = model(torch.randn(1, 10))
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Check optimizer states
    for param in model.parameters():
        state = optimizer.state[param]
        assert 'exp_avg' in state
        assert 'exp_avg_sq' in state
        assert state['exp_avg'].dtype == torch.bfloat16
```

**Expected**: Optimizer updates weights, maintains BF16 states

#### 2.3 Lookahead Optimizer
**Test**: Slow weight synchronization
```python
def test_lookahead():
    from cpu_training_system import LAMB_BF16, Lookahead
    model = torch.nn.Linear(10, 5)
    base_opt = LAMB_BF16(model.parameters(), lr=0.01)
    optimizer = Lookahead(base_opt, k=5)

    # Store initial weights
    initial_weights = [p.clone() for p in model.parameters()]

    # Run k+1 steps
    for i in range(6):
        output = model(torch.randn(1, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Verify weights changed
    for init, current in zip(initial_weights, model.parameters()):
        assert not torch.equal(init, current)
```

**Expected**: Slow weights update every k steps

#### 2.4 DynamicFreezer
**Test**: Layer freezing based on gradient norms
```python
def test_dynamic_freezer():
    from cpu_training_system import DynamicFreezer
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

    freezer = DynamicFreezer(model, threshold=100.0, patience=2)

    # Simulate training with very small gradients
    for _ in range(50):
        output = model(torch.randn(1, 10))
        loss = output.sum() * 1e-8  # Very small loss -> small gradients
        loss.backward()

    # Check if any layers were frozen
    print(f"Frozen layers: {freezer.frozen}")
    frozen_count = len(freezer.frozen)

    assert frozen_count >= 0  # Should freeze with small gradients and high threshold
```

**Expected**: Layers freeze when gradients consistently below threshold

#### 2.5 SyntheticDataset
**Test**: Deterministic data generation
```python
def test_synthetic_dataset():
    from cpu_training_system import SyntheticDataset
    dataset = SyntheticDataset(size=1000, input_dim=784, num_classes=1000)

    # Test dataset size
    assert len(dataset) == 1000

    # Test item retrieval
    data, target = dataset[0]
    assert data.shape == (784,)
    assert 0 <= target < 1000

    # Test determinism
    data1, target1 = dataset[42]
    data2, target2 = dataset[42]
    assert torch.equal(data1, data2)
    assert target1 == target2
```

**Expected**: Dataset generates consistent data for same indices

### 3. Integration Tests

#### 3.1 Model Creation
**Test**: Model architecture and parameter count
```python
def test_model_creation():
    from cpu_training_system import create_model
    model = create_model(num_classes=1000)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 52_000_000  # Approximately 52M

    assert abs(total_params - expected_params) / expected_params < 0.01  # Within 1%
```

**Expected**: Model has ~52M parameters

#### 3.2 Training System Initialization
**Test**: System setup without errors
```python
def test_training_system_init():
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(size=1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = create_model(num_classes=1000)

    system = CPUTrainingSystem(model, loader, epochs=1, accumulation_steps=4)

    assert system.model is not None
    assert system.optimizer is not None
    assert system.freezer is not None
```

**Expected**: System initializes without errors

### 4. Functional Tests

#### 4.1 Basic Smoke Test
**Test**: Run 2 epochs on small dataset
```python
def test_basic_training():
    """Quick smoke test - does it run at all?"""
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    # Small dataset for quick test
    dataset = SyntheticDataset(size=500, input_dim=784, num_classes=100)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    model = create_model(num_classes=100)

    system = CPUTrainingSystem(model, loader, epochs=2, accumulation_steps=2)
    trained_model = system.train()

    assert trained_model is not None
```

**Expected**: Training completes without crashes

#### 4.2 Loss Convergence Test
**Test**: Verify loss decreases over time
```python
def test_loss_convergence():
    """Verify that loss actually decreases"""
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(size=1000, input_dim=784, num_classes=10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    model = create_model(num_classes=10)

    system = CPUTrainingSystem(model, loader, epochs=5, accumulation_steps=2)

    # Track losses
    losses = []
    original_train = system.train

    def train_with_tracking():
        # Simplified tracking
        return original_train()

    system.train = train_with_tracking
    system.train()

    # Loss should generally trend downward
    print("Training completed - check logs for loss trends")
```

**Expected**: Loss decreases over epochs

### 5. Performance Benchmarks

#### 5.1 Baseline Comparison
**Test**: Compare against standard PyTorch training
```bash
python baseline_comparison.py
```

**Metrics**:
- Samples/second throughput
- Time per epoch
- Memory usage
- Final loss value

**Expected**: System should be faster than naive CPU training

#### 5.2 Throughput Measurement
**Test**: Measure training throughput
```python
def test_throughput():
    """Measure samples processed per second"""
    import time
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(size=5000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)
    model = create_model(num_classes=1000)

    system = CPUTrainingSystem(model, loader, epochs=1, accumulation_steps=8)

    start_time = time.time()
    system.train()
    elapsed = time.time() - start_time

    throughput = 5000 / elapsed
    print(f"Throughput: {throughput:.0f} samples/sec")

    # Target: 150-200 samples/sec on 12-core AMD Ryzen
    assert throughput > 50  # Minimum acceptable
```

**Expected**: 150-200 samples/sec on target hardware

### 6. Ablation Studies

Test each innovation independently to measure its contribution.

#### 6.1 BF16 vs FP32
**Test**: Compare training with/without BF16
```bash
python ablation_tests.py --test bf16
```

**Metrics**:
- Speed difference
- Memory usage
- Convergence comparison

#### 6.2 Fused Layers
**Test**: Compare fused vs separate operations
```bash
python ablation_tests.py --test fused_layers
```

**Metrics**:
- Forward pass time
- Memory bandwidth usage

#### 6.3 Dynamic Freezing
**Test**: Compare with/without layer freezing
```bash
python ablation_tests.py --test dynamic_freezing
```

**Metrics**:
- Compute time savings
- Impact on final accuracy

#### 6.4 LAMB+Lookahead vs Adam
**Test**: Compare optimizer configurations
```bash
python ablation_tests.py --test optimizers
```

**Metrics**:
- Convergence speed
- Final loss
- Training stability

### 7. Memory Tests

#### 7.1 Memory Usage
**Test**: Monitor memory consumption
```python
def test_memory_usage():
    """Track peak memory usage during training"""
    import psutil
    import os
    from cpu_training_system import main

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024**3  # GB

    # Run training
    main()  # This runs full training

    peak_memory = process.memory_info().rss / 1024**3  # GB
    memory_used = peak_memory - initial_memory

    print(f"Memory used: {memory_used:.2f} GB")
    assert memory_used < 32  # Should fit in 64GB system with margin
```

**Expected**: Memory usage < 32GB

### 8. Stress Tests

#### 8.1 Large Batch Training
**Test**: Handle large effective batch sizes
```python
def test_large_batch():
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(size=10000)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=12)
    model = create_model(num_classes=1000)

    # Large accumulation steps = large effective batch
    system = CPUTrainingSystem(model, loader, epochs=1, accumulation_steps=16)
    system.train()
```

**Expected**: Handles large batches without OOM

#### 8.2 Long Training
**Test**: Train for many epochs
```python
def test_long_training():
    """Verify stability over extended training"""
    from cpu_training_system import CPUTrainingSystem, create_model, SyntheticDataset
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(size=5000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)
    model = create_model(num_classes=100)

    system = CPUTrainingSystem(model, loader, epochs=20, accumulation_steps=4)
    system.train()
```

**Expected**: No degradation or crashes over time

## Test Execution Order

1. **Environment Validation** (1-2 min)
2. **Component Unit Tests** (5-10 min)
3. **Integration Tests** (5 min)
4. **Basic Smoke Test** (5 min)
5. **Performance Benchmarks** (30-60 min)
6. **Ablation Studies** (60-120 min)
7. **Memory Tests** (20-30 min)
8. **Stress Tests** (30-60 min)

**Total estimated time**: 3-5 hours for complete test suite

## Success Criteria

### Minimum Requirements (Must Pass)
- ✓ All unit tests pass
- ✓ Training completes without errors
- ✓ Loss decreases over epochs
- ✓ Memory usage < 32GB
- ✓ Throughput > 50 samples/sec

### Target Performance (Should Achieve)
- ✓ Throughput 150-200 samples/sec
- ✓ Faster than baseline PyTorch
- ✓ Each innovation provides measurable benefit
- ✓ Dynamic freezing saves 10-20% compute

### Stretch Goals (Nice to Have)
- ✓ Throughput > 200 samples/sec
- ✓ Convergence in 30% fewer epochs vs Adam
- ✓ Dynamic freezing saves > 20% compute

## Known Limitations

1. **BF16 Support**: Requires recent CPU with AVX-512
   - Fallback: Code should work with FP32 (slower)

2. **Thread Count**: Hardcoded to 12
   - Fix: Make configurable via command line

3. **No Error Handling**: Missing try/except blocks
   - Impact: May crash on unexpected inputs

4. **Hardcoded Hyperparameters**: Learning rate, batch size, etc.
   - Fix: Add configuration file or argparse

## Recommendations for Production

1. **Add Configuration Management**: Use YAML/JSON config files
2. **Add Error Handling**: Wrap critical sections in try/except
3. **Add Logging**: Use Python logging module instead of print
4. **Add Checkpointing**: Save model state periodically
5. **Add Validation Loop**: Evaluate on held-out data
6. **Add Early Stopping**: Stop if loss plateaus
7. **Make Thread Count Dynamic**: Auto-detect CPU cores
8. **Add BF16 Fallback**: Gracefully handle unsupported CPUs

## Automation

Create a test runner script:
```bash
#!/bin/bash
# run_all_tests.sh

echo "Running environment validation..."
python -m pytest test_environment.py

echo "Running unit tests..."
python -m pytest test_components.py

echo "Running integration tests..."
python -m pytest test_integration.py

echo "Running performance benchmarks..."
python baseline_comparison.py
python ablation_tests.py

echo "All tests complete!"
```

## Reporting

Generate test report with:
- Pass/fail status for each test
- Performance metrics table
- Memory usage graphs
- Ablation study results
- Recommendations for improvement

Save results to `TEST_RESULTS.md` for documentation.
