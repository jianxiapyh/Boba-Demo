# Garden asset provenance and license

The runtime Garden Gaussian is downloaded locally from the official Graphdeco
pre-trained-model archive. It is not redistributed by this repository.

- Model project: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- Model source: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip>
- Capture source: <https://jonbarron.info/mipnerf360/>
- Graphdeco license: <https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md>

The Graphdeco model and software are provided for non-commercial research and
evaluation under their license. Users running the Garden fetch command are
responsible for complying with the upstream model and dataset terms.

The committed calibration, deterministic cleanup parameters, generated
collision proxy description, and integration code are part of Boba-Demo; the
large source and processed PLY files remain under the ignored `data/garden/`
directory.
