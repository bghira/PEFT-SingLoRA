"""
PEFT-SingLoRA: Single Low-Rank Adaptation for PEFT

A parameter-efficient fine-tuning method that uses a single low-rank matrix
instead of two, reducing parameters while maintaining performance.
"""

from .layer import SingLoRALayer, Linear
from .config import SingLoRAConfig, setup_singlora

__version__ = "0.1.0"
__all__ = ["SingLoRALayer", "Linear", "SingLoRAConfig", "setup_singlora"]

# Convenience function to set up SingLoRA globally
def setup_singlora():
    """
    Automatically register SingLoRA layers with PEFT.
    Call this once at the beginning of your script.
    """
    import warnings
    try:
        from peft import LoraConfig
        import torch.nn as nn
        
        # Register SingLoRA for Linear layers
        custom_module_mapping = {nn.Linear: Linear}
        
        # Monkey-patch LoraConfig to auto-register SingLoRA
        _original_init = LoraConfig.__init__
        
        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            self._register_custom_module(custom_module_mapping)
            
        LoraConfig.__init__ = _patched_init
        
    except ImportError:
        warnings.warn(
            "PEFT not installed. Please install it with: pip install peft>=0.7.0"
        )