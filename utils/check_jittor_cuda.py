"""
Quick Jittor/CUDA sanity check.
"""

import jittor as jt


def main():
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
