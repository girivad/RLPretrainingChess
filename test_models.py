import torch
import pytest
from model import GPT, SFGPT, MTPGPT, GPTConfig, SFGPTConfig, MTPGPTConfig

# I had Claude generate a quick set of tests to validate GPT, SFGPT and MTPGPT on single and multi-GPU setups.

# Test fixtures
@pytest.fixture
def small_config():
    """Tiny model for fast testing"""
    return dict(
        n_layer=2,
        n_head=2,
        n_embd=32,
        hidden_dim=64,
        block_size=16,
        vocab_size=29,
        dropout=0.0,
        bias=False
    )

@pytest.fixture
def batch():
    """Small batch of data"""
    B, T = 2, 8
    X = torch.randint(0, 29, (B, T))
    Y = torch.randint(0, 29, (B, T))
    return X, Y

# ============================================================================
# CRITICAL TESTS - These would have caught our bugs
# ============================================================================

def test_gpt_forward_shapes(small_config, batch):
    """Test basic forward pass shapes"""
    config = GPTConfig(**small_config)
    model = GPT(config)
    X, Y = batch
    
    # Full forward
    logits, loss, loss_tensor = model(X, Y)
    assert logits.shape == (X.shape[0], X.shape[1], config.vocab_size)
    assert loss.ndim == 0  # scalar
    assert loss_tensor.shape == (2,)
    assert not torch.isnan(loss)

def test_sfgpt_forward_shapes(small_config, batch):
    """Test SFGPT forward - would have caught the embeddings bug"""
    config = SFGPTConfig(**small_config, n_slayer=1, lamda=0.5)
    model = SFGPT(config)
    X, Y = batch
    
    # This would fail with the embeddings bug
    logits, loss, loss_tensor = model(X, Y)
    assert logits.shape == (X.shape[0], X.shape[1], config.vocab_size)
    assert loss_tensor.shape == (6,)  # SFGPT has more loss terms
    assert not torch.isnan(loss)

def test_mtpgpt_forward_shapes(small_config, batch):
    """Test MTPGPT forward"""
    config = MTPGPTConfig(**small_config, k=2, discount_rate=0.99)
    model = MTPGPT(config)
    X, Y = batch
    
    logits, loss, loss_tensor = model(X, Y, k=0)
    assert logits.shape == (X.shape[0], X.shape[1], config.vocab_size)
    assert not torch.isnan(loss)

def test_probing_interface(small_config, batch):
    """Test that probing works for all models - would have caught end_layer bug"""
    X, _ = batch
    
    for model_class, config_class in [
        (GPT, GPTConfig),
        (SFGPT, lambda **kw: SFGPTConfig(**kw, n_slayer=1)),
        (MTPGPT, lambda **kw: MTPGPTConfig(**kw, k=2))
    ]:
        config = config_class(**small_config)
        model = model_class(config)
        
        # Extract layer 1 representations
        hidden = model(X, end_layer=1)
        assert hidden.shape == (X.shape[0], X.shape[1], config.n_embd)
        
        # Extract all layer representations
        for layer in range(config.n_layer + 1):
            hidden = model(X, end_layer=layer)
            # Should not crash

def test_mtpgpt_single_gpu(small_config, batch):
    """Test MTPGPT training in single-GPU mode - would have caught hasattr bug"""
    from torch.amp import GradScaler
    from contextlib import nullcontext
    
    config = MTPGPTConfig(**small_config, k=2)
    model = MTPGPT(config)
    X, Y = batch
    
    scaler = GradScaler('cuda', enabled=False)
    
    # This would crash with the hasattr bug
    loss, loss_tensor = model.compute_gradient(
        X, Y, 
        gradient_accumulation_steps=1,
        ctx=nullcontext(),
        scaler=scaler,
        ddp_model=None  # Single GPU
    )
    
    assert not torch.isnan(loss)

def test_inference_mode(small_config):
    """Test inference mode (no targets)"""
    config = GPTConfig(**small_config)
    model = GPT(config)
    model.eval()
    
    X = torch.randint(0, 29, (1, 4))
    
    with torch.no_grad():
        logits, loss = model(X, targets=None)
        assert logits.shape == (1, 1, config.vocab_size)  # Only last token
        assert loss is None

def test_evaluation_mode(small_config, batch):
    """Test evaluation mode (returns loss without extras)"""
    config = GPTConfig(**small_config)
    model = GPT(config)
    X, Y = batch
    
    logits, loss = model(X, Y, evaluate=True)
    assert logits.shape == (X.shape[0], X.shape[1], config.vocab_size)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

# ============================================================================
# OPTIONAL: Quick integration test
# ============================================================================

def test_one_training_step(small_config, batch):
    """Smoke test: can we do one training step?"""
    config = GPTConfig(**small_config)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X, Y = batch
    
    # Forward
    logits, loss, _ = model(X, Y)
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Should complete without error
    assert True

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
        test_gpt_forward_shapes(small_config, batch)
        print("✓ GPT forward")
        
        test_sfgpt_forward_shapes(small_config, batch)
        print("✓ SFGPT forward")
        
        test_mtpgpt_forward_shapes(small_config, batch)
        print("✓ MTPGPT forward")
        
        test_probing_interface(small_config, batch)
        print("✓ Probing interface")
        
        test_mtpgpt_single_gpu(small_config, batch)
        print("✓ MTPGPT single GPU")
        
        test_inference_mode(small_config)
        print("✓ Inference mode")
        
        test_evaluation_mode(small_config, batch)
        print("✓ Evaluation mode")
        
        test_one_training_step(small_config, batch)
        print("✓ Training step")
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)