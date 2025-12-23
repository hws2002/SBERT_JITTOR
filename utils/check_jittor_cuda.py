"""
Quick Jittor/CUDA sanity check.
"""

import jittor as jt


def check_torch_transformers():
    try:
        import torch
        import transformers
    except Exception as exc:
        print("PyTorch/Transformers import failed:", exc)
        return False

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Transformers version:", transformers.__version__)
    return True


def main():
    check_torch_transformers()
    print("Jittor version:", jt.__version__)
    print("Has CUDA:", jt.has_cuda)
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        x = jt.randn((1024, 1024))
        y = x @ x
        y.sync()
        print("CUDA compute OK")
    else:
        print("CUDA not available")


if __name__ == "__main__":
    main()
