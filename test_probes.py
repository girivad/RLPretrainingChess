import torch
import pytest
import numpy as np


from model import GPT, SFGPT, MTPGPT, ProbeHead, ProbeWrapper, GPTConfig, SFGPTConfig, MTPGPTConfig, ProbeHeadConfig
from probe import get_batch, get_dtype, evaluate, accuracy

# Test fixtures
@pytest.fixture
def small_base_config():
    """Tiny model for fast testing"""
    return dict(
        n_layer=2,
        n_head=2,
        n_embd=32,
        hidden_dim=64,
        block_size=16,
        vocab_size=1969,
        dropout=0.0,
        bias=False
    )

@pytest.fixture
def batch():
    """Small batch of data"""
    B, T = 2, 8
    V, C = 1969, 3
    X = torch.randint(0, V, (B, T))
    Y = torch.randint(0, C, (B, T))
    return X, Y

@pytest.fixture
def create_data(V, C, block_size, num_train=100, num_val=20):
    """Create random training and validation data"""
    train_data = np.random.randint(0, V, size=(num_train, block_size - 1), dtype=get_dtype(V))
    train_data = np.concatenate([np.zeros((num_train, 1), dtype=get_dtype(V)), train_data], axis=1)
    val_data = np.random.randint(0, V, size=(num_val, block_size - 1), dtype=get_dtype(V))
    val_data = np.concatenate([np.zeros((num_val, 1), dtype=get_dtype(V)), val_data], axis=1)

    train_outcomes = np.random.randint(0, C, size=(num_train, 1)).astype(np.float16).repeat(block_size, axis=1)
    train_outcomes[:, -1:] = np.nan  # let the last token have no outcome
    val_outcomes = np.random.randint(0, C, size=(num_val, 1)).astype(np.float16).repeat(block_size, axis=1)
    val_outcomes[:, -1:] = np.nan  # let the last token have no outcome

    return (train_data, train_outcomes), (val_data, val_outcomes)

# ============================================================================

def test_dataloader(data, V, C, batch_size, block_size, start_probe_turn, end_probe_turn):
    (train_data, train_outcomes), (val_data, val_outcomes) = data

    X, Y = get_batch("train")
    assert X.shape == (batch_size, block_size)
    assert Y.shape == (batch_size, block_size)

    assert X.max() < V and X.min() >= 0
    assert Y.max() < C and Y.min() >= 0

    assert np.all(np.isnan(Y[:, :start_probe_turn]))
    assert np.all(~np.isnan(Y[:, start_probe_turn:end_probe_turn]))
    assert np.all(np.isnan(Y[:, end_probe_turn:]))

    assert np.all(np.isnan(Y[:, -1]))  # last token has no outcome for testing purposes (simulates padding-induced nan labels)
    assert np.all(X[:, 0] == 0)  # BOS token

def test_probe_wrapper(small_config, data, C):
    (train_data, train_outcomes), (val_data, val_outcomes) = data
    start_probe_turn = 20, end_probe_turn = 60

    X, Y = get_batch("train")

    config = GPTConfig(**small_config)
    model = GPT(config)
    model = ProbeWrapper(
        base_model = model,
        probe_head_config = ProbeHeadConfig(
            architecture = "linear",
            task = "classification",
            n_classes = C,
            embd_dim = config.n_embd,
        )
    )

    probe_outs, total_loss, loss_components = model(X, targets = Y)
    assert all(probe_outs[layer_idx].size() == (X.shape[0], X.shape[1], C) for layer_idx in range(config.n_layer))

    # Loss and Backprop Properties to test for:
    # No NaNs: Ensure that the total_loss does not contain NaN values
    assert not torch.any(torch.isnan(total_loss))
    # White Player Probes vs Black Player Probes: Should both be included in the total_loss (can be verified by checking gradients after backprop)
    total_loss.backward()
    assert torch.all(
        [
            model.white_turn_probes[layer_idx].grad is not None 
                and 
            torch.any(model.white_turn_probes[layer_idx].grad != 0) 
            for layer_idx in range(config.n_layer)
        ]
    )
    assert torch.all(
        [
            model.black_turn_probes[layer_idx].grad is not None 
                and 
            torch.any(model.black_turn_probes[layer_idx].grad != 0) 
            for layer_idx in range(config.n_layer)
        ]
    )    
    # Base Model Parameters should be frozen: Verify that parameters of the base GPT model do not receive gradients during backpropagation
    assert all(param.grad is None or torch.all(param.grad == 0) for param in model.base_model.parameters())
    model.reset_grad()
    # Total Loss Calculation: Should be the sum of all individual probe losses
    assert torch.sum(loss_components) == total_loss.item()

    # Logit Properties to test for:
    # Shape Consistency: Verify that the shapes of probe outputs match expected dimensions
    assert all(probe_outs[layer_idx].size() == (X.shape[0], X.shape[1], C) for layer_idx in range(config.n_layer))
    # Logits should be nan where targets are nan
    for layer_idx in range(config.n_layer):
        probe_logits = probe_outs[layer_idx]
        assert torch.all(torch.isnan(probe_logits[torch.isnan(Y)]) )

def test_eval(small_config, data, C):
    (train_data, train_outcomes), (val_data, val_outcomes) = data
    start_probe_turn = 20, end_probe_turn = 60

    config = GPTConfig(**small_config)
    model = GPT(config)
    model = ProbeWrapper(
        base_model = model,
        probe_head_config = ProbeHeadConfig(
            architecture = "linear",
            task = "classification",
            n_classes = C,
            embd_dim = config.n_embd,
        )
    )

    probe_outs, _, _ = model(X, targets = Y)
    assert all(probe_outs[layer_idx].size() == (X.shape[0], X.shape[1], C) for layer_idx in range(config.n_layer))

    n_layer = config.n_layer
    batch_size = 32
    eval_iters = 10

    eval_metrics = evaluate()

    assert all([~torch.any(torch.isnan(eval_metric)) for eval_metric in eval_metrics.values()])
    print("Metrics:", eval_metrics)

# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    # Can run without pytest for quick checks
    import sys
    
    small_config = dict(
        n_layer=2, n_head=2, n_embd=32, hidden_dim=64,
        block_size=16, vocab_size=29, dropout=0.0, bias=False
    )
    X = torch.randint(0, 29, (2, 8))
    Y = torch.randint(0, 29, (2, 8))
    batch = (X, Y)
    
    print("Running minimal tests...")
    try:
        test_dataloader(
            batch, 
            V=1969, 
            C=3, 
            batch_size=2, 
            block_size=30, 
            start_probe_turn=4, 
            end_probe_turn=40
        )
        print("✓ Dataloader Tested")

        test_probe_wrapper(small_config, batch, C=3)
        print("✓ Probe Wrapper Tested")

        test_eval(small_config, batch, C=3)
        print("✓ Evaluation Tested")
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)