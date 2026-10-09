"""hessboost's native extension built with NVIDIA CUDA (``hessboost-runtime-cuda``).

Private, with no API of its own: ``import hessboost`` loads
``_hessboost_runtime_cuda._native`` in place of the extension the
``hessboost`` package ships whenever this distribution is installed (as
``hessboost[cuda]`` installs it), provided it is the same release.
"""
