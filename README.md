# TIDE: Time Derivative Diffusion for Deep Learning on Graphs

TIDE is described in ["TIDE: Time Derivative Diffusion for Deep Learning on Graphs"](https://arxiv.org/pdf/2212.02483.pdf), by

Maximilian Krahn,
[Maysam Behmanesh](https://maysambehmanesh.github.io/),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)


Abstract

A prominent paradigm for graph neural networks is based on the message passing framework. In this framework, information communication is realized only between neighboring nodes. The challenge of approaches that use this paradigm is to ensure efficient and accurate long distance communication between nodes, as deep convolutional networks are prone to over-smoothing. In this paper, we present a novel method based on time derivative graph diffusion (TIDE), with a learnable time parameter. Our approach allows to adapt the spatial extent of diffusion across different tasks and network channels, thus enabling medium and long-distance communication efficiently. Furthermore, we show that our architecture directly enables local message passing and thus inherits from the expressive power of local message passing approaches. We show that on widely used graph benchmarks we achieve comparable performance and on a synthetic mesh dataset we outperform state-of-the-art methods like GCN or GRAND by a significant margin.


![image](https://user-images.githubusercontent.com/77163765/205882230-a29ee9a7-a4bf-4f34-8dcc-e5d8530bdb5b.png)

## Requirements
- ogb>=1.3.3
- torch>=1.10.0
- torch-geometric>=2.0.4

## Reference

```
@misc{https://doi.org/10.48550/arxiv.2212.02483,
  doi = {10.48550/ARXIV.2212.02483},
  url = {https://arxiv.org/abs/2212.02483},
  author = {Krahn, Maximilian and Behmanesh, Maysam and Ovsjanikov, Maks},
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), Social and Information Networks (cs.SI), FOS: Computer and information sciences, FOS: Computer and information sciences},
   title = {TIDE: Time Derivative Diffusion for Deep Learning on Graphs},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
