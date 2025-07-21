# PEFT-SingLoRA

[![PyPI version](https://badge.fury.io/py/peft-singlora.svg)](https://badge.fury.io/py/peft-singlora)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

SingLoRA (Single Low-Rank Adaptation) is an efficient alternative to traditional LoRA that uses a single low-rank matrix instead of two, reducing parameters while maintaining performance. This package provides a PEFT-compatible implementation of SingLoRA.

## Key Features

- 🚀 **50% fewer parameters** than standard LoRA
- 🔧 **Fully compatible with PEFT** ecosystem
- 📊 **Mathematically equivalent** adaptation capability
- 🎯 **Easy integration** with existing PEFT workflows

## Installation

```bash
pip install peft-singlora
```

## Quick Start

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft_singlora import setup_singlora

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-name")

# Setup SingLoRA (this registers it with PEFT)
setup_singlora()

# Configure LoRA as usual - it will use SingLoRA under the hood
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# Create PEFT model - will automatically use SingLoRA for linear layers
peft_model = get_peft_model(model, config)

# Train as usual!
```

## How It Works

Traditional LoRA uses two matrices (A and B) for the low-rank decomposition:
```
W = W_0 + BA
```

SingLoRA uses a single matrix A with a symmetric decomposition:
```
W = W_0 + α/r * A @ A^T
```

This reduces trainable parameters from `2 * d * r` to `d * r` while maintaining the same expressive power.

## Advanced Usage

### Custom Configuration

```python
from peft_singlora import SingLoRAConfig

config = SingLoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    ramp_up_steps=1000,  # Gradually increase adaptation strength
)
```

### Manual Integration

```python
import torch.nn as nn
from peft_singlora import Linear as SingLoRALinear

# Register custom module mapping
custom_module_mapping = {nn.Linear: SingLoRALinear}
config._register_custom_module(custom_module_mapping)
```

## Examples

Check out the [examples/](https://github.com/bghira/PEFT-SingLoRA/tree/main/examples) directory for:
- Basic usage with different model architectures
- Fine-tuning examples with real datasets
- Performance comparisons with standard LoRA

## Citation

If you use SingLoRA in your research, please cite:

```bibtex
@article{singlora2024,
  title={SingLoRA: Rethinking Low-Rank Adaptation for Parameter-Efficient Fine-Tuning},
  author={...},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.