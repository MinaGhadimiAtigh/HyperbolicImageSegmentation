# Hyperbolic Image Segmentation, CVPR 2022

This is the implementation of paper [Hyperbolic Image Segmentation (CVPR 2022)](https://arxiv.org/pdf/2203.05898.pdf).

![Figure 1](assets/HIS.jpeg)

## Repository structure 

- <b>assets </b>: images and stuff
- <b>datasets </b>: contains integer to class dictionaries, and JSON files that contain the hierarchies used.
- <b>hesp </b>: the actual code containing layers, models, losses, etc.
- <b>samples </b>: helper files, bash scripts, and train.py

**Code is not complete yet.**

## How to use the code?

For installation, first run <code> pip install -e .</code>  to register the package.

Then, run <code>sh requirements.sh</code> to install the requirements. 

The code needs Tensorflow 1, 
the experiments are performed using Tensorflow 1.14. The tensorflow installed by the script is tensorflow-cpu. Change the commands to install tensorflow on GPU.

To train a model, use this code in <code>samples</code> directory.

 <code>python train.py --mode segmenter --batch_size 5 --dataset coco --geometry hyperbolic --dim 256 --c 0.1 --freeze_bn --train --test --backbone_init Path_to_resnet/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt --output_stride 16 --segmenter_ident check</code>

The code will train and test a hyperbolic model using coco stuff dataset, with batch size 5, curvature 0.1, freeze batch
normalization, output stride 16. The result will be saved in a folder named 
<code>poincare-hesp/save/segmenter/hierarchical_coco_d256_hyperbolic_c0.1_os16_resnet_v2_101_bs5_lr0.001_fbnTrue_fbbFalse_check</code> in the samples directory.

To get the dataset tfrecord files and resnet pretrained weights, use [this link](https://drive.google.com/drive/folders/1AggSC8fKCgRsfYjTjVRjoOfa85g2KBBu?usp=sharing).

## Citation
Please consider citing this work using this BibTex entry,

```bibtex
@article{ghadimiatigh2022hyperbolic,
  title={Hyperbolic Image Segmentation},
  author={GhadimiAtigh, Mina and Schoep, Julian and Acar, Erman and van Noord, Nanne and Mettes, Pascal},
  journal={arXiv preprint arXiv:2203.05898},
  year={2022}
}
```
