# TIDE: Time Derivative Diffusion for Deep Learning on Graphs

TIDE is described in ["TIDE: Time Derivative Diffusion for Deep Learning on Graphs"](https://arxiv.org/pdf/2212.02483.pdf), by

[Maysam Behmanesh*](https://maysambehmanesh.github.io/),
[Maximilian Krahn*](https://scholar.google.com/citations?user=Dg5q7-QAAAAJ),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)


Abstract

A prominent paradigm for graph neural networks is based on the message-passing framework. In this framework, information communication is realized only between neighboring nodes. The challenge of approaches that use this paradigm is to ensure efficient and accurate long-distance communication between nodes, as deep convolutional networks are prone to over-smoothing. In this paper, we present a novel method based on time derivative graph diffusion (TIDE), with a learnable time parameter. Our approach allows us to adapt the spatial extent of diffusion across different tasks and network channels, thus enabling medium and long-distance communication efficiently. Furthermore, we show that our architecture directly enables local message-passing and thus inherits from the expressive power of local message-passing approaches. We show that on widely used graph benchmarks we achieve comparable performance and on a synthetic mesh dataset we outperform state-of-the-art methods like GCN or GRAND by a significant margin.

![TIDE-framework](https://github.com/maysambehmanesh/TIDE/assets/77163765/a1bb8f6a-9e29-4388-9c09-00c11d186ea3)

## 
Please take note that Equation (10) has been simplified similarly to the following equation, as you can find in the implementation.

![eq10](https://github.com/maysambehmanesh/TIDE/assets/77163765/c0e6904c-e4c6-466a-9705-c448acfc5c72)


## Requirements
- ogb>=1.3.3
- torch>=1.10.0
- torch-geometric>=2.0.4

## Citation

```
@InProceedings{pmlr-v202-behmanesh23a,
  title = 	 {{TIDE}: Time Derivative Diffusion for Deep Learning on Graphs},
  author =       {Behmanesh, Maysam and Krahn, Maximilian and Ovsjanikov, Maks},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {2015--2030},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/behmanesh23a/behmanesh23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/behmanesh23a.html},
}
```

## Reference
Sharp et al. [DiffusionNet: Discretization Agnostic Learning on Surfaces](https://github.com/nmwsharp/diffusion-net). TOG 2022.
