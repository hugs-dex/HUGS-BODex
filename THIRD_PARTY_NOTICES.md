# Third-party notices and release gates

NVIDIA notices, `LICENSE` and `LICENSE_ASSETS` are preserved. NVIDIA License section
3.3 restricts use to non-commercial research/evaluation.

| Component | Fixed source | Status |
| --- | --- | --- |
| pytorch_kinematics | https://github.com/DexGrasp-TH/pytorch_kinematics at `b7d87b1d01bd7db91ca17f5269abe6d365f3e646` | MIT in submodule LICENSE.txt; anonymously fetched |
| utils_python | https://github.com/Mingrui-Yu/utils_python at `d28c65780e43339d497f3d06986e6d3dc5d80484` | Retained; anonymously fetched; this version has no license file, terms need owner confirmation |
| MANO/manopth | https://mano.is.tue.mpg.de/ and https://github.com/hassony2/manopth | Separate code/model terms; models not included |
| Robot assets | `src/curobo/content/assets/robot` | Existing notices retained; LICENSE_ASSETS is not a blanket grant for added Shadow/Leap-SP derivatives |
| Object assets | https://huggingface.co/datasets/JiayiChenPKU/BODex | External inputs; archive version/digest/terms still require verification |

The inventory checker records bytes/SHA-256 of the ten-mode robot/config closure.
This candidate stores assets as ordinary Git blobs, without private LFS endpoints.
Publication remains gated on author/source/license review and a clean fetch of the
final public asset distribution. URL accessibility does not prove redistribution rights.
