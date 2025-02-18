# Low-Resolution Self-Attention for Semantic Segmentation

This is the official repository for **Low-Resolution Self-Attention for Semantic Segmentation**. We introduce a self-attention mechanism (LRSA) that computes attention in a fixed low-resolution space regardless of the input image size. This design drastically reduces computational cost while still capturing global context. Based on LRSA, we build the Low-Resolution Transformer (LRFormer) with an encoder-decoder architecture, and we provide several variants (T/S/B/L/XL). In addition, we release enhanced versions (LRFormer+) that combine the LRFormer encoder with a query-based decoder (e.g., Mask2Former) for even stronger performance.

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.7.0+
- mmcv >= 2.0.0rc4
- mmdet >= 3.0
- Other dependencies are listed in `requirements.txt`

Tested with `pytorch 2.2.1, torchvision 0.17.1, mmcv 2.1.0, mmdet 3.3, and cuda 11.8`.

---

## Installation

You can follow the installation of `mmsegmentation`. For simplicity,
run the following commands:

```bash
pip install -r requirements.txt
python setup.py build develop 
```

---

## Introduction

Semantic segmentation requires both fine-grained detail and global context. While traditional high-resolution self-attention effectively captures local details, its computational complexity is a major bottleneck. Our proposed Low-Resolution Self-Attention (LRSA) computes self-attention in a fixed, low-resolution space—regardless of input size—thereby significantly reducing FLOPs. To compensate for any loss in local detail, we further integrate 3×3 depth-wise convolutions. Based on LRSA, we design LRFormer, a vision transformer backbone tailored for semantic segmentation. Extensive experiments on ADE20K, COCO-Stuff, and Cityscapes demonstrate that LRFormer consistently outperforms state-of-the-art models with lower computational cost.

For even higher performance, we also offer an enhanced version (LRFormer+) that couples the LRFormer encoder with a query-based decoder (e.g., Mask2Former), yielding superior segmentation results.

---

## Pretrained Models & Results

We provide pretrained models on three popular semantic segmentation datasets. The following tables summarize the performance (FLOPs, parameter count, and mIoU) of our LRFormer and LRFormer+ variants.

### ADE20K (Validation)

| Variant         | FLOPs | Params | mIoU  |
|-----------------|:-----:|:------:|:-----:|
| LRFormer-T      |  17G  |  13M   | 46.7% |
| LRFormer-S      |  40G  |  32M   | 50.0% |
| LRFormer-B      |  75G  |  69M   | 51.0% |
| LRFormer-L      | 183G  | 113M   | 52.6% |
| LRFormer-L†     | 183G  | 113M   | 54.2% |

### COCO-Stuff (Validation)

| Variant         | FLOPs | Params | mIoU  |
|-----------------|:-----:|:------:|:-----:|
| LRFormer-T      |  17G  |  13M   | 43.9% |
| LRFormer-S      |  40G  |  32M   | 46.4% |
| LRFormer-B      |  75G  |  69M   | 47.2% |
| LRFormer-L      | 122G  | 113M   | 47.9% |

### Cityscapes (Validation)

*FLOPs are calculated for an input size of 1024×2048.*

| Variant         |   FLOPs    | Params | mIoU  |
|-----------------|:----------:|:------:|:-----:|
| LRFormer-T      |   122G     |  13M   | 80.7% |
| LRFormer-S      |   295G     |  32M   | 81.9% |
| LRFormer-B      |   555G     |  67M   | 83.0% |
| LRFormer-L      |   908G     | 111M   | 83.2% |

---

### Enhanced Versions with Query-based Decoders (LRFormer+)

On ADE20K, we further improve segmentation by coupling our LRFormer encoder with a query-based decoder. The following table reports the performance of the LRFormer+ series:

| Variant          | FLOPs | Params | mIoU  |
|------------------|:-----:|:------:|:-----:|
| LRFormer-T+      |  53G  |  31M   | 49.4% |
| LRFormer-S+      |  70G  |  48M   | 51.3% |
| LRFormer-B+      |  94G  |  80M   | 53.7% |
| LRFormer-L+†     | 192G  | 119M   | 55.8% |
| LRFormer-XL+†    | 365G  | 205M   | 58.1% |

*“†” indicates results with ImageNet-22K pretraining and larger input size (640×640).*

All models will be available for access on Google Drive and Baidu Pan.

Google Drive:


---

## Training

To train LRFormer (e.g., the LRFormer-S variant) in a distributed manner on 8 GPUs, run:

```bash
bash tools/dist_train.sh configs/mask2former/lrformer-s-plus-160k_ade20k-512x512.py 8
# bash tools/dist_train.sh $CONFIG_PATH $NUM_GPUS
```

## Testing


```bash
bash tools/dist_test.sh configs_local/lrformer/lrformer_l_ade20k_160k.py model_release/lrformer-l-160k_ade20k_52.6.pth 1
# bash tools/dist_test.sh $CONFIG_PATH $MODEL_PATH $NUM_GPUS
```


### Other Notes

If you meet any problems, please do not hesitate to contact us.
Issues and discussions are welcome in the repository!
You can also contact us via sending messages to this email: wuyuhuan@mail.nankai.edu.cn


### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Non-Commercial use only. Any commercial use should get formal permission first.

