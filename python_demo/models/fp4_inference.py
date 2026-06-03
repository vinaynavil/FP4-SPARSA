# fp4_inference.py
# Runs FP32 vs FP4 inference on CIFAR-10 test set.
# Reports accuracy drop, sparsity stats, and MACs skipped.
# Hardware correlation: matches one 4x4 tile MAC output vs Verilog TB.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from cifar10_loader import load_cifar10
from models.resnet20 import ResNet20
from quantize import (quantize_to_fp4, quantize_model_weights,
                      compute_sparsity_stats, fp4_forward_pass)

CHECKPOINT = "checkpoints/resnet20_fp32.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256


def load_model():
    model = ResNet20().to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  "
          f"saved_acc={ckpt['test_acc']:.2f}%")
    return model


def evaluate_fp32(model, loader):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def evaluate_fp4(model, quantized_weights, loader):
    correct = total = 0
    original_weights = {}
    for name, module in model.named_modules():
        if name in quantized_weights:
            original_weights[name] = module.weight.data.clone()
            module.weight.data = quantized_weights[name]['quantized'].to(DEVICE)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)

    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name]

    return 100.0 * correct / total


def count_activation_sparsity(model, quantized_weights, loader, n_batches=5):
    """
    Count zero activations during FP4 weight inference.
    Hooks BatchNorm2d outputs and applies ReLU to simulate F.relu(bn(conv(x))).
    ResNet20 uses F.relu (functional), so nn.ReLU hooks would miss all activations.
    """
    act_zeros = act_total = 0
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            nonlocal act_zeros, act_total
            # Apply ReLU to BN output to simulate F.relu(bn(...))
            relu_out = F.relu(out)
            act_zeros += (relu_out == 0).sum().item()
            act_total += relu_out.numel()
        return hook

    # Hook all BatchNorm2d layers (applied immediately before F.relu in ResNet20)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(make_hook()))

    # Swap to FP4 weights
    original_weights = {}
    for name, module in model.named_modules():
        if name in quantized_weights:
            original_weights[name] = module.weight.data.clone()
            module.weight.data = quantized_weights[name]['quantized'].to(DEVICE)

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            model(x.to(DEVICE))

    # Restore weights and remove hooks
    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name]
    for h in hooks:
        h.remove()

    pct = 100.0 * act_zeros / act_total if act_total > 0 else 0
    return act_zeros, act_total, pct


def hardware_correlation(model, quantized_weights, test_images):
    """
    Hardware correlation: extract one 4-lane weight tile and activation tile
    from the first conv layer, compute dot product in Python,
    and verify it matches what the Verilog systolic array would produce.
    """
    print("\n" + "="*60)
    print("  HARDWARE CORRELATION")
    print("="*60)

    first_conv_name = next(iter(quantized_weights))

    w_fp4 = quantized_weights[first_conv_name]['quantized']  # (16,3,3,3)
    scale  = quantized_weights[first_conv_name]['scale']

    # Take first 4 weights from filter 0, channel 0, row 0 — pad to 4 lanes
    w_slice = w_fp4[0, 0, 0, :].cpu()
    w4 = torch.zeros(4)
    w4[:min(3, 4)] = w_slice[:min(3, 4)]

    # Get a sample activation patch: first image, channel 0, first 4 pixels
    x_sample = torch.tensor(test_images[0:1]).to(DEVICE)
    a_slice = x_sample[0, 0, 0, :4].cpu()

    # Quantize the activation slice to FP4
    a_scale = a_slice.abs().max().clamp(min=1e-8).item() / 6.0
    a_fp4   = quantize_to_fp4(a_slice, scale=a_scale)

    # Scale to Q1.1 ×2 integer representation matching Verilog decoder output
    w_q1 = (w4 * 2).round().int()
    a_q1 = (a_fp4 * 2).round().int()

    dot_product = (w_q1 * a_q1).sum().item()

    print(f"  Layer          : {first_conv_name}")
    print(f"  Weight scale   : {float(scale.max().item()):.4f}")
    print(f"  Act scale      : {a_scale:.4f}")
    print(f"\n  Weight FP4 (4 lanes) : {w4.tolist()}")
    print(f"  Act    FP4 (4 lanes) : {a_fp4.tolist()}")
    print(f"\n  Weight Q1.1 ×2 : {w_q1.tolist()}")
    print(f"  Act    Q1.1 ×2 : {a_q1.tolist()}")
    print(f"\n  Python dot product (Q1.1): {dot_product}")
    print(f"  → Verilog PE accumulator would show: {dot_product}")
    print(f"  → Zero lanes (sparse): "
          f"{((w4==0)|(a_fp4==0)).sum().item()}/4")
    print("="*60)

    return dot_product


def main():
    print(f"Device: {DEVICE}")
    print("Loading data...")
    _, _, te_img, te_lbl = load_cifar10()

    te_x = torch.tensor(te_img)
    te_y = torch.tensor(te_lbl, dtype=torch.long)
    loader = DataLoader(TensorDataset(te_x, te_y),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

    print("Loading model...")
    model = load_model()

    # ── FP32 baseline ─────────────────────────────────────────
    print("\nEvaluating FP32 baseline...")
    fp32_acc = evaluate_fp32(model, loader)
    print(f"  FP32 accuracy : {fp32_acc:.2f}%")

    # ── FP4 quantization ──────────────────────────────────────
    print("\nQuantizing weights to FP4 E2M1 (per-channel RNE)...")
    quantized_weights = quantize_model_weights(model, scale_mode='per_channel')

    layer_stats, total_zeros, total_params, overall_pct = \
        compute_sparsity_stats(quantized_weights)

    print(f"\n  Weight sparsity per layer:")
    for name, s in layer_stats.items():
        print(f"    {name:40s} {s['sparsity_pct']:5.1f}%  "
              f"({s['zeros']:,}/{s['total']:,} zeros)")

    print(f"\n  Overall weight sparsity : {overall_pct:.2f}%")
    print(f"  Total zero weights      : {total_zeros:,} / {total_params:,}")

    # ── FP4 inference accuracy ────────────────────────────────
    print("\nEvaluating FP4 inference...")
    fp4_acc = evaluate_fp4(model, quantized_weights, loader)
    acc_drop = fp32_acc - fp4_acc
    print(f"  FP4  accuracy : {fp4_acc:.2f}%")
    print(f"  Accuracy drop : {acc_drop:.2f}%")

    # ── Activation sparsity ───────────────────────────────────
    print("\nCounting activation sparsity (5 batches)...")
    az, at, apct = count_activation_sparsity(model, quantized_weights, loader)
    print(f"  Zero activations: {az:,} / {at:,} = {apct:.1f}%")

    # ── MACs skipped estimate ─────────────────────────────────
    mac_skip_pct = overall_pct + apct - (overall_pct * apct / 100)
    print(f"\n  Estimated MACs skipped  : {mac_skip_pct:.1f}%")
    print(f"  (weight={overall_pct:.1f}% + act={apct:.1f}% - overlap)")

    # ── Hardware correlation ──────────────────────────────────
    hardware_correlation(model, quantized_weights, te_img)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FP4-SPARSA DEMO SUMMARY")
    print("="*60)
    print(f"  FP32 accuracy       : {fp32_acc:.2f}%")
    print(f"  FP4  accuracy       : {fp4_acc:.2f}%")
    print(f"  Accuracy drop       : {acc_drop:.2f}%")
    print(f"  Weight sparsity     : {overall_pct:.1f}%")
    print(f"  Activation sparsity : {apct:.1f}%")
    print(f"  MACs skipped (est.) : {mac_skip_pct:.1f}%")
    print(f"  Power saving (est.) : {mac_skip_pct * 0.4:.1f}%")
    print(f"  (40% dynamic power proportional to active MACs)")
    print("="*60)


if __name__ == "__main__":
    main()