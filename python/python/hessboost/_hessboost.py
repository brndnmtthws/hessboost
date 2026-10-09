"""The native extension, as the package's modules import it.

Two builds of the extension exist: this package's own (``hessboost._native``,
without CUDA) and the CUDA runtime's (``_hessboost_runtime_cuda._native``,
from the ``hessboost-runtime-cuda`` distribution that ``hessboost[cuda]``
installs on Linux). This module replaces itself in :data:`sys.modules` with
the CUDA runtime's when that distribution is installed and with this
package's otherwise, so ``from hessboost import _hessboost`` binds the
extension itself (typed by ``_hessboost.pyi``).

A CUDA runtime of another release raises :class:`ImportError` rather than
load: its extension need not match this release's Python layer (a partial
upgrade leaves one behind).
"""

import sys
from importlib import import_module, metadata

_CUDA_RUNTIME = "hessboost-runtime-cuda"


def _extension() -> str:
    """The name of the extension module to load."""
    try:
        runtime = metadata.version(_CUDA_RUNTIME)
    except metadata.PackageNotFoundError:
        return "hessboost._native"
    release = metadata.version("hessboost")
    if runtime != release:
        raise ImportError(
            f"hessboost {release} cannot load {_CUDA_RUNTIME} {runtime}: the CUDA runtime "
            f'must be the same release. Install "hessboost[cuda]=={release}", or uninstall '
            f"{_CUDA_RUNTIME} to run without CUDA."
        )
    return "_hessboost_runtime_cuda._native"


sys.modules[__name__] = import_module(_extension())
