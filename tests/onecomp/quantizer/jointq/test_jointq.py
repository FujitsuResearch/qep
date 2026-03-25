"""Tests for the JointQ quantizer implementation."""

import sys
import os
import torch
import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from onecomp.quantizer.jointq._jointq import JointQ

    HAS_JOINTQ = True
except ImportError:
    HAS_JOINTQ = False

from onecomp.quantizer._quantizer import QuantizationResult
from test_module import BaseQuantizeSpec


@pytest.mark.skipif(not HAS_JOINTQ, reason="jointq package not installed")
class TestJointQ(BaseQuantizeSpec):
    """Test cases for JointQ quantization.

    Note: JointQ requires the external `jointq` package.
    JointQ returns a plain tensor (auto-wrapped as QuantizationResult).
    """

    __test__ = True
    quantizer_cls = JointQ if HAS_JOINTQ else None
    result_cls = QuantizationResult
    default_parameter_for_test = {
        "bits": 1,
        "symmetric": False,
        "group_size": 1,
        "batch_size": 1,
    }
    boundary_parameters = [
        # bits: int >= 1 (validated by validate_params), no explicit upper
        {"bits": 1},  # bits lower boundary
        {"bits": 32},  # bits large value (no explicit upper bound)
        # symmetric: bool
        {"symmetric": True},
        {"symmetric": False},
        # group_size: int >= 1 (validated by validate_params), no explicit upper
        {"group_size": 1},  # group_size lower boundary
        {"group_size": 1024},  # group_size large value (no explicit upper bound)
        # batch_size: int >= 0 (validated by validate_params), no explicit upper
        {"batch_size": 0},  # batch_size lower boundary
        {"batch_size": 10000},  # batch_size large value (no explicit upper bound)
        # log_level: int in 0..2 (validated by validate_params)
        {"log_level": 0},  # log_level lower boundary
        {"log_level": 2},  # log_level upper boundary
        # ils_enabled: bool
        {"ils_enabled": True},
        {"ils_enabled": False},
        # ILS sub-params: lower boundaries (no explicit upper)
        {"ils_num_iterations": 1},  # ils_num_iterations lower boundary
        {"ils_num_clones": 1},  # ils_num_clones lower boundary
        {"ils_num_channels": 1},  # ils_num_channels lower boundary
        # ILS sub-params: large values (no explicit upper)
        {"ils_num_iterations": 100},  # ils_num_iterations large value (no explicit upper bound)
        {"ils_num_clones": 100},  # ils_num_clones large value (no explicit upper bound)
        {"ils_num_channels": 10000},  # ils_num_channels large value (no explicit upper bound)
        # ILS combo: lower boundaries
        {"ils_enabled": True, "ils_num_iterations": 1, "ils_num_clones": 1, "ils_num_channels": 1},
        # ils_num_channels: None is also valid (auto-detect)
        {
            "ils_enabled": True,
            "ils_num_iterations": 1,
            "ils_num_clones": 1,
            "ils_num_channels": None,
        },
        # all class defaults
        {
            "bits": 4,
            "symmetric": False,
            "group_size": 128,
            "batch_size": 4096,
            "log_level": 1,
            "ils_enabled": True,
            "ils_num_iterations": 10,
            "ils_num_clones": 8,
            "ils_num_channels": 512,
        },
        # all minimum
        {
            "bits": 1,
            "symmetric": False,
            "group_size": 1,
            "batch_size": 0,
            "log_level": 0,
            "ils_enabled": False,
            "ils_num_iterations": 1,
            "ils_num_clones": 1,
            "ils_num_channels": 1,
        },
        # all maximum
        {
            "bits": 32,
            "symmetric": True,
            "group_size": 1024,
            "batch_size": 10000,
            "log_level": 2,
            "ils_enabled": True,
            "ils_num_iterations": 100,
            "ils_num_clones": 100,
            "ils_num_channels": 10000,
        },
    ]
    abnormal_parameters = [
        {"bits": 0},  # below lower boundary (bits >= 1)
        {"group_size": 0},  # below lower boundary (group_size >= 1)
        {"batch_size": -1},  # below lower boundary (batch_size >= 0)
        {"log_level": -1},  # below lower boundary (log_level in 0..2)
        {"log_level": 3},  # above upper boundary (log_level in 0..2)
        {
            "ils_enabled": True,
            "ils_num_iterations": 0,
        },  # below lower boundary (ils_num_iterations >= 1)
        {"ils_enabled": True, "ils_num_clones": 0},  # below lower boundary (ils_num_clones >= 1)
        {
            "ils_enabled": True,
            "ils_num_channels": 0,
        },  # below lower boundary (ils_num_channels >= 1 or None)
    ]

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and devices of quantize_layer outputs."""
        assert isinstance(result, self.result_cls)
        assert hasattr(result, "dequantized_weight")
        assert isinstance(result.dequantized_weight, torch.Tensor)
        assert result.dequantized_weight.shape == layer.weight.shape
        assert result.dequantized_weight.device == torch.device("cpu")

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        assert torch.equal(r1.dequantized_weight, r2.dequantized_weight)

    def check_quantize_error(self, error, max_error):
        """Validate that quantization error is within tolerance."""
        assert error < 0.4
        assert max_error < 1.71

    def check_forward_error(
        self,
        error_original_vs_dequantized,
        error_dequantized_vs_applied,
        max_error_dequantized_vs_applied,
    ):
        """Validate forward errors."""
        print(
            "[JointQ forward error] "
            f"original_vs_jointq(rel={error_original_vs_dequantized:.8f}), "
            f"jointq_vs_jointql(max={max_error_dequantized_vs_applied:.8f}), "
            f"jointq_vs_jointql(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"JointQ dequantized vs applied max error too large: "
            f"{max_error_dequantized_vs_applied}"
        )

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        module.weight.data = result.dequantized_weight.to(device)

    def test_forward_error(self, helper):
        """Skip forward error test (no inference layer support)."""
        pytest.skip("JointQ does not support create_inference_layer")
