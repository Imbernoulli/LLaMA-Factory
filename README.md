### What I Changed

I updated three parts to make things run faster and use less memory:

1.  **Flash Attention 3**
2.  **Gradient Checkpointing**
3.  **Liger Kernel**

### How to Run

You can use the environment in `run4.sh`.

If you set up your own environment, you need to do two things:

1.  **Install Liger Kernel:** You must install this manually from here:
    [https://github.com/Comet0322/Liger-Kernel](https://github.com/Comet0322/Liger-Kernel)

2.  **Download Flash Attention 3:** If you are in a place with an internet connection, first run the code below to download and cache Flash Attention 3.

    ```python
    import torch
    from kernels import get_kernel
    vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")
    ```