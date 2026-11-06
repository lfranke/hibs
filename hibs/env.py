import os
import random

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUPY_NVRTC_USE_PCH", "1")


def configure(tf32=False, seed=0):
    import numpy as np
    import torch

    # math attention only: cuDNN attention shifts LighterGlue match scores and breaks poses on TUM
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def use_backbone_cwd():
    from hibs import I3DGS

    os.chdir(I3DGS)


def disable_xformers_attention():
    from depth_anything_v2.dinov2_layers import attention

    attention.XFORMERS_AVAILABLE = False
