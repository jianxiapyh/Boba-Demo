# Boba-Demo Vendored `gsplat`

- Fork source: the `Boba_Batched` branch of Boba-Latest
- Fork source commit: `99e50055a60a4bc7e5022abba1a938bf386b273d`
- Upstream repository: `https://github.com/nerfstudio-project/gsplat`
- Upstream tag: `v1.5.3`
- Upstream commit: `937e29912570c372bed6747a5c9bf85fed877bae`
- Demo path: `gaussian_splatting/submodules/gsplat`

The runtime Python, CUDA, C++, and required GLM header files are copied exactly
from that Boba-Batched commit. Documentation, examples, profiling utilities,
CI configuration, caches, generated package metadata, and GLM's tests/docs are
excluded from the demo distribution. The upstream Apache-2.0 license is in
`LICENSE`; GLM's MIT/Happy Bunny license text is retained in
`gsplat/cuda/csrc/third_party/glm/manual.md`.

Boba-Demo imports this source tree directly and JIT-compiles its CUDA extension
when first needed. It deliberately does not install an editable package into
the shared `phystwin` environment. The demo uses this fork's standard
`rasterization()` entry point for its one-instance/two-camera render while
validating `rasterization_shared_template()` as the custom-fork marker.
