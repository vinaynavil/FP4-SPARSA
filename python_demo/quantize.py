# quantize.py
# FP4 E2M1 quantizer matching Verilog fp4_decoder exactly.
#
# FP4 E2M1 format: [3]=sign  [2:1]=exponent  [0]=mantissa  bias=1
# Representable values (post-FTZ): 0, ±1, ±1.5, ±2, ±3, ±4, ±6
# Subnormals (±0.5) flushed to zero — matches Verilog FTZ NOR gate.
#
# Quantization: round-to-nearest-even (RNE) then clip to FP4 range.
# Sparsity counters match Verilog v15c lane-wise semantics.

import torch
import numpy as np

# ── FP4 E2M1 representable values (post-FTZ) ─────────────────
FP4_VALUES = torch.tensor([
    0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,        # positive
   -1.0,-1.5,-2.0,-3.0,-4.0,-6.0               # negative
], dtype=torch.float32)

FP4_MAX = 6.0
FP4_MIN = -6.0


def quantize_to_fp4(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Quantize a float tensor to the nearest FP4 E2M1 value.
    Uses round-to-nearest-even (RNE) via nearest-neighbour search.
    Subnormals flushed to zero (FTZ) — matches Verilog fp4_decoder.

    Args:
        x     : input tensor (any shape), float32
        scale : per-tensor scale factor (x_scaled = x / scale)
    Returns:
        Quantized tensor in float32, values from FP4_VALUES only.
    """
    x_scaled = x / scale
    x_clipped = x_scaled.clamp(FP4_MIN, FP4_MAX)

    # Nearest-neighbour search over FP4 representable values
    fp4 = FP4_VALUES.to(x.device)
    # x_clipped: (...) → (..., 1), fp4: (13,) → broadcast
    dist = (x_clipped.unsqueeze(-1) - fp4).abs()
    idx  = dist.argmin(dim=-1)
    quantized = fp4[idx]

    return quantized * scale


def quantize_model_weights(model, scale_mode='per_tensor'):
    """
    Returns a dict of quantized weight tensors for all Conv2d/Linear layers.
    Does NOT modify model in-place — returns separate quantized copies.

    scale_mode: 'per_tensor' (simpler) or 'per_channel' (NVFP4 style)
    """
    quantized_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w = module.weight.data.float()
            if scale_mode == 'per_channel':
                # Scale per output channel — matches NVIDIA transformer-engine
                if w.dim() == 4:  # Conv: (out, in, kH, kW)
                    scale = w.abs().amax(dim=(1,2,3), keepdim=True).clamp(min=1e-8) / FP4_MAX
                else:             # Linear: (out, in)
                    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP4_MAX
            else:
                scale = w.abs().max().clamp(min=1e-8).item() / FP4_MAX

            quantized_weights[name] = {
                'quantized': quantize_to_fp4(w, scale),
                'scale'    : scale,
                'original' : w,
            }
    return quantized_weights


def count_sparsity(tensor: torch.Tensor):
    """
    Count zero elements in a quantized tensor.
    Returns: (zero_count, total_count, sparsity_percent)
    """
    zeros = (tensor == 0.0).sum().item()
    total = tensor.numel()
    return zeros, total, 100.0 * zeros / total


def compute_sparsity_stats(quantized_weights: dict):
    """
    Compute sparsity stats across all quantized weight tensors.
    Matches v15c semantics: reports zero lanes as fraction of total MACs.
    """
    total_zeros = total_params = 0
    layer_stats = {}

    for name, data in quantized_weights.items():
        z, t, pct = count_sparsity(data['quantized'])
        layer_stats[name] = {'zeros': z, 'total': t, 'sparsity_pct': pct}
        total_zeros  += z
        total_params += t

    overall_pct = 100.0 * total_zeros / total_params if total_params > 0 else 0.0
    return layer_stats, total_zeros, total_params, overall_pct


def fp4_forward_pass(model, x: torch.Tensor, quantized_weights: dict):
    """
    Run inference substituting FP4 quantized weights into the model.
    Activations remain FP32 (weight-only quantization).
    Returns logits.
    """
    # Temporarily swap weights
    original_weights = {}
    for name, module in model.named_modules():
        if name in quantized_weights:
            original_weights[name] = module.weight.data.clone()
            module.weight.data = quantized_weights[name]['quantized'].to(x.device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    # Restore original weights
    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name]

    return logits


if __name__ == "__main__":
    print("Testing FP4 E2M1 quantizer...")

    # Test quantization of known values
    test_vals = torch.tensor([-6.5, -6.0, -4.0, -3.0, -1.5, -1.0,
                               -0.5,  0.0,  0.5,  1.0,  1.5,
                                2.0,  3.0,  4.0,  6.0,  6.5])
    quantized = quantize_to_fp4(test_vals)
    print(f"{'Input':>8} → {'FP4':>6}")
    print("-" * 20)
    for v, q in zip(test_vals.tolist(), quantized.tolist()):
        print(f"{v:8.1f} → {q:6.1f}")

    # Verify sparsity counting
    sparse_test = torch.tensor([0.0, 1.0, 0.0, -1.5, 0.0, 6.0])
    z, t, pct = count_sparsity(sparse_test)
    print(f"\nSparsity test: {z}/{t} zeros = {pct:.1f}%")
    assert pct == 50.0, "Sparsity count mismatch"

    # Verify FTZ: 0.5 → 0.0
    ftz_test = quantize_to_fp4(torch.tensor([0.4, 0.5, 0.6, -0.5]))
    print(f"\nFTZ test (0.4,0.5,0.6,-0.5) → {ftz_test.tolist()}")
    assert ftz_test[1].item() == 0.0, "FTZ failed: 0.5 should flush to 0"
    assert ftz_test[2].item() == 1.0, "0.6 should round to 1.0"

    print("\nquantize.py OK")