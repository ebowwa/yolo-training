# Research Papers Reference

Annotated bibliography of research papers integrated into the edge-training repository.

---

## Transformer Optimization

### Layer-wise Pruning of Transformer Attention Heads
- **arXiv**: [2201.08071](https://arxiv.org/abs/2201.08071)
- **Year**: 2022
- **Key Contribution**: Proportional reduction in computation by pruning attention heads layer-by-layer in "All-attention" Transformers
- **Implementation**: `service/layers/pruning.py`
- **Relevant Classes**: `AttentionHeadPruner`, `PrunedMultiHeadAttention`, `HeadImportanceScorer`

**Key Insights**:
- Many attention heads are redundant and can be removed with minimal performance loss
- Layer-wise pruning allows proportional FLOPs reduction (unlike standard attention pruning)
- Three stability techniques: gradual pruning, importance smoothing, and knowledge distillation

---

### RT-DETR: DETRs Beat YOLOs on Real-time Object Detection
- **arXiv**: [2304.08069](https://arxiv.org/abs/2304.08069)  
- **Year**: 2023
- **Key Contribution**: First real-time end-to-end transformer detector, eliminating NMS
- **Implementation**: `service/rfdetr_service.py`
- **Relevant Classes**: `RFDETRService`, `RFDETRDetection`

**Key Insights**:
- Hybrid encoder with decoupled intra-scale and cross-scale fusion
- Uncertainty-minimal query selection for high-quality decoder inputs
- Flexible speed tuning via decoder layer count (no retraining needed)
- Achieves 60.5 mAP on COCO with deterministic latency

---

### Deformable DETR: Deformable Transformers for End-to-End Object Detection
- **arXiv**: [2010.04159](https://arxiv.org/abs/2010.04159)
- **Year**: 2020
- **Key Contribution**: Deformable attention for efficient multi-scale feature processing
- **Implementation**: Informs design of `service/optimization.py`

**Key Insights**:
- Attention focuses on a small set of key sampling points around a reference
- Combines sparse spatial sampling of deformable convolution with Transformer relation modeling
- 10x faster convergence than vanilla DETR
- Better performance on small objects

---

## Efficient Architectures

### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **arXiv**: [1704.04861](https://arxiv.org/abs/1704.04861)
- **Authors**: Andrew G. Howard et al. (Google)
- **Year**: 2017
- **Key Contribution**: Depthwise separable convolutions for mobile efficiency
- **Implementation**: `service/layers/depthwise_separable.py`
- **Relevant Classes**: `DepthwiseSeparableConv2d`, `InvertedResidual`, `DepthwiseConv2d`

**Key Insights**:
- Factorizes standard convolution into depthwise + pointwise
- ~8-9x reduction in computation with minimal accuracy loss
- Width multiplier (α) for uniform network thinning
- Resolution multiplier (ρ) for input resolution scaling

**Computational Savings**:
```
Standard Conv:     D_k² × M × N × D_f²
Depthwise Sep:     D_k² × M × D_f² + M × N × D_f²
Reduction Ratio:   1/N + 1/D_k²
```

---

### Once-for-All: Train One Network and Specialize it for Efficient Deployment
- **arXiv**: [1908.09791](https://arxiv.org/abs/1908.09791)
- **Venue**: ICLR 2020
- **Key Contribution**: Single supernet supporting >10^19 sub-networks without retraining
- **Implementation**: `service/layers/ofa.py`
- **Relevant Classes**: `ElasticKernel`, `ElasticWidth`, `ElasticBlock`, `OFASubNetworkExtractor`

**Key Insights**:
- Decouples training from architecture search
- Progressive shrinking: depth → width → kernel size
- Hardware-aware sub-network selection
- Up to 4% better than MobileNetV3 on ImageNet

**Training Phases**:
1. Full network warmup
2. Elastic kernel (3/5/7)
3. Elastic depth
4. Elastic width

---

## Knowledge Distillation

### Patient Knowledge Distillation for BERT Model Compression
- **arXiv**: [2012.06785](https://arxiv.org/abs/2012.06785)
- **Year**: 2019
- **Key Contribution**: Multi-layer distillation from teacher's intermediate layers
- **Implementation**: `service/layers/distillation.py`
- **Relevant Classes**: `PatientDistillationLoss`, `SoftLabelLoss`, `FeatureMatchingLoss`

**Key Insights**:
- Standard distillation only uses final layer outputs
- PKD matches multiple intermediate layers for richer supervision
- Two variants: PKD-Last (last K layers) and PKD-Skip (every K-th layer)
- Improves training efficiency and model accuracy

**PKD Variants** (12-layer teacher → 6-layer student):
| Variant | Student Layer | Teacher Layer |
|---------|---------------|---------------|
| PKD-Last | 1,2,3,4,5,6 | 7,8,9,10,11,12 |
| PKD-Skip | 1,2,3,4,5,6 | 2,4,6,8,10,12 |

---

## Edge Deployment

### Edge ML Optimization Survey
- **Topics**: Quantization, pruning, distillation, hardware acceleration
- **Key Techniques**:
  - **Quantization**: FP32 → INT8/INT4 for faster inference
  - **Pruning**: Remove unnecessary weights/neurons/heads
  - **Distillation**: Transfer knowledge to smaller models
  - **Hardware**: TensorRT, NCNN, CoreML, TFLite optimizations

**Implementation Status**:
- ✅ INT8 quantization via `ExportService`
- ✅ Attention head pruning via `pruning.py`
- ✅ Knowledge distillation via `distillation.py`
- ✅ Multiple export formats (NCNN, CoreML, TFLite, ONNX)

---

## Related Work (Not Yet Implemented)

| Paper | arXiv | Topic | Status |
|-------|-------|-------|--------|
| DistilBERT | 1910.01108 | BERT distillation | Concepts in `distillation.py` |
| EfficientNet | 1905.11946 | Compound scaling | Consider for future |
| TinyBERT | 1909.10351 | Transformer distillation | Concepts in `distillation.py` |
| Lottery Ticket | 1803.03635 | Sparse networks | Consider for future |

---

## Citation

If you use these implementations, please cite the original papers:

```bibtex
@article{howard2017mobilenets,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Howard, Andrew G and others},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}

@inproceedings{cai2020once,
  title={Once-for-All: Train One Network and Specialize it for Efficient Deployment},
  author={Cai, Han and Gan, Chuang and Wang, Tianzhe and Zhang, Zhekai and Han, Song},
  booktitle={ICLR},
  year={2020}
}

@article{sun2019patient,
  title={Patient Knowledge Distillation for BERT Model Compression},
  author={Sun, Siqi and others},
  journal={arXiv preprint arXiv:1908.09355},
  year={2019}
}

@article{lv2023detrs,
  title={DETRs Beat YOLOs on Real-time Object Detection},
  author={Lv, Wenyu and others},
  journal={arXiv preprint arXiv:2304.08069},
  year={2023}
}
```
