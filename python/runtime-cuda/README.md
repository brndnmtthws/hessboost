# hessboost-runtime-cuda

The [hessboost](https://pypi.org/project/hessboost/) extension built with
NVIDIA CUDA, for Linux. Install it through hessboost's `cuda` extra, which
pins the matching release:

```sh
uv add 'hessboost[cuda]'        # or: pip install 'hessboost[cuda]'
```

`import hessboost` then loads this extension in place of its own:
`device="cuda"` (or `"cuda:<ordinal>"`) trains and `Booster.to_gpu("cuda")`
predicts on an NVIDIA GPU, bit for bit as on the CPU. Everything else,
wgpu included, works as in the plain `hessboost` package. This package has
no Python API of its own. It must come from the same release as
`hessboost`, which refuses to import beside one of another release.

Running it needs no CUDA toolkit, only an NVIDIA GPU of compute capability
7.5 (Turing) or newer and a driver supporting CUDA 12.8 or newer. Wheels
cover x86_64 and aarch64 glibc Linux (manylinux_2_28): one `abi3` wheel for
CPython 3.11 and newer and one for free-threaded CPython 3.14t. Building
from the source distribution needs Rust 1.93 or newer, a C compiler,
CUDA 13's `cuda.h` and `curand.h` (`CUDA_HOME` or `CUDA_TOOLKIT_PATH`), and
libclang.

See hessboost's [GPU documentation](https://github.com/brndnmtthws/hessboost/tree/main/python#gpu-training-and-prediction).

## License

Apache-2.0.
