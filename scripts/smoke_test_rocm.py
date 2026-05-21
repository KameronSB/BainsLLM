from __future__ import annotations

import sys
import time

import torch


def main() -> None:
    print("=" * 80)
    print("BainsLLM ROCm / PyTorch smoke test")
    print("=" * 80)

    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version:   {torch.__version__}")
    print(f"HIP version:       {torch.version.hip}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() returned False. "
            "This means PyTorch is not seeing your ROCm GPU."
        )

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    print(f"Device count:      {torch.cuda.device_count()}")
    print(f"Current device:    {torch.cuda.current_device()}")
    print(f"Device name:       {device_name}")

    # Small GPU tensor check
    x = torch.randn(1024, 1024, device=device)
    print(f"Tensor device:     {x.device}")

    if x.device.type != "cuda":
        raise RuntimeError(f"Expected tensor on cuda, got {x.device}")

    # Matmul test
    print("\nRunning matmul test...")
    a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    start = time.time()

    c = a @ b

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Matmul output device: {c.device}")
    print(f"Matmul output dtype:   {c.dtype}")
    print(f"Matmul mean:           {c.float().mean().item():.6f}")
    print(f"Matmul time:           {elapsed:.4f}s")

    if c.device.type != "cuda":
        raise RuntimeError("Matmul output is not on GPU.")

    # Backward pass test
    print("\nRunning tiny backward pass...")
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
    ).to(device)

    inp = torch.randn(64, 512, device=device)
    target = torch.randn(64, 512, device=device)

    out = model(inp)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    first_param = next(model.parameters())

    print(f"Loss:                {loss.item():.6f}")
    print(f"Model param device:  {first_param.device}")
    print(f"Grad exists:         {first_param.grad is not None}")

    if first_param.device.type != "cuda":
        raise RuntimeError("Model parameters are not on GPU.")

    if first_param.grad is None:
        raise RuntimeError("Backward pass did not produce gradients.")

    print("\nSmoke test passed: PyTorch is running on the ROCm GPU, not CPU.")
    print("=" * 80)


if __name__ == "__main__":
    main()
