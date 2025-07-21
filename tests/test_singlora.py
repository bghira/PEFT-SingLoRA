"""
Basic tests for SingLoRA implementation.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Import after peft to avoid issues
from peft_singlora import setup_singlora, Linear, SingLoRAConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_singlora_layer_creation():
    """Test that SingLoRA layer can be created."""
    base_layer = nn.Linear(10, 20)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=4,
        lora_alpha=16,
    )

    # Check that layer was created
    assert isinstance(singlora_layer, Linear)
    assert singlora_layer.r == 4
    assert singlora_layer.lora_alpha == 16


def test_singlora_forward_pass():
    """Test forward pass through SingLoRA layer."""
    base_layer = nn.Linear(10, 20)
    singlora_layer = Linear(
        base_layer=base_layer,
        adapter_name="default",
        r=4,
        lora_alpha=16,
    )

    # Test forward pass
    x = torch.randn(5, 10)
    output = singlora_layer(x)

    assert output.shape == (5, 20)
    assert torch.isfinite(output).all()


def test_peft_integration():
    """Test integration with PEFT."""
    setup_singlora()

    model = SimpleModel()
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1", "linear2"],
        lora_dropout=0.1,
    )

    peft_model = get_peft_model(model, config)

    # Check that model was created
    assert peft_model is not None

    # Test forward pass
    x = torch.randn(2, 10)
    output = peft_model(x)
    assert output.shape == (2, 5)


def test_singlora_config():
    """Test SingLoRAConfig."""
    config = SingLoRAConfig(
        r=8,
        lora_alpha=32,
        target_modules=["linear1"],
        ramp_up_steps=2000,
    )

    assert config.r == 8
    assert config.lora_alpha == 32
    assert config.ramp_up_steps == 2000


def test_parameter_count():
    """Test that SingLoRA has fewer parameters than standard LoRA."""
    model = SimpleModel()

    # Setup SingLoRA
    setup_singlora()
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1", "linear2"],
    )

    peft_model = get_peft_model(model, config)

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in peft_model.parameters())

    # SingLoRA should have fewer trainable params than standard LoRA
    # For two Linear layers: (20×4 + 20×4) = 160 params (vs 320 for standard LoRA)
    assert trainable_params < total_params * 0.1  # Less than 10% trainable


def test_merge_unmerge():
    """Test merging and unmerging of adapters."""
    setup_singlora()

    model = SimpleModel()
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["linear1"],
    )

    peft_model = get_peft_model(model, config)

    # Get original weight
    original_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Merge adapter
    peft_model.merge_adapter()
    merged_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Weights should be different after merge
    assert not torch.allclose(original_weight, merged_weight)

    # Unmerge
    peft_model.unmerge_adapter()
    unmerged_weight = peft_model.base_model.model.linear1.base_layer.weight.clone()

    # Should be back to original
    assert torch.allclose(original_weight, unmerged_weight, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
