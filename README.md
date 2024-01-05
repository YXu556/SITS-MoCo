# SITS-MoCo

PyTorch implementation of  ["Self-supervised pre-training for large-scale crop mapping using Sentinel-2 time series"](http://dx.doi.org/10.1016/j.isprsjprs.2023.12.005)

<img src="png/Figure_3_1.png" title="" alt="" data-align="center">

**Abstract:** Large-scale crop mapping is essential for various agricultural applications, such as yield prediction and agricultural resource management. The current most advanced techniques for crop mapping utilize deep learning (DL) models on satellite imagery time series (SITS). Despite advancements, the efficacy of DL-based crop mapping methods is impeded by the arduous task of acquiring crop-type labels and the extensive pre-processing required on satellite data. To address these issues, we proposed a Transformer-based DL model and a self-supervised pre-training framework for the label-scarce crop mapping task. Specifically, we first developed a Transformer-based Spectral Temporal Network (STNet) which is designed to extract task-informative features from time-series remote sensing (RS) imagery via the self-attention mechanism. A self-supervised pre-training strategy, namely SITS-MoCo, was then proposed to learn robust and generalizable representations from time-series RS imagery that is invariant to spectral noise, temporal shift, and irregular-length data. To evaluate the proposed framework, experiments were conducted using Sentinel-2 time series and high-confident Cropland Data Layer (CDL) reference data on six geographically scattered study sites across the United States from 2019 to 2021. The experimental results demonstrated that the framework had superior performance in comparison to other advanced DL models and self-supervised pre-training techniques. The pre-training strategy was proven to effectively alleviate the need for complex data pre-processing and training labels for the downstream crop mapping task. Overall, this research presented a novel pipeline for improving model performance on lar  ge-scale crop mapping with limited labels and provided a viable solution to efficiently exploit available satellite data that can be easily adapted to other large-area classification tasks.

## Requirements

* Pytorch 3.8.12, PyTorch 1.11.0, and more in `environment.yml`

## Usage

Setup conda environment and activate

```
conda env create -f environment.yml
conda activate py38
```

Set `DATAPATH` in `main_tscls.py` or `main_moco.py` to your data path. 



Example: pre-train model using SITS-MoCo

```
python main_moco.py transformer --rc --use-doy --useall --mlp
```

Train STNet with pre-trained model

```
python main_tscls.py stnet --rc --pretrained checkpoints/pretrained/MoCoV2_TRSF_doy/model_best.pth
```

## Reference

In case you find SITS-MoCo or the code useful, please consider citing our paper using the following BibTex entry:

```
@article{xu_self-supervised_2024,
	title = {Self-supervised pre-training for large-scale crop mapping using Sentinel-2 time series},
	volume = {207},
	issn = {0924-2716},
	doi = {10.1016/j.isprsjprs.2023.12.005},
	pages = {312--325},
	journaltitle = {{ISPRS} Journal of Photogrammetry and Remote Sensing},
	shortjournal = {{ISPRS} Journal of Photogrammetry and Remote Sensing},
	author = {Xu, Yijia and Ma, Yuchi and Zhang, Zhou},
}

```

## Credits

- The implementation of MoCo is based on [the official implementation](https://github.com/facebookresearch/moco)

- The Sentinel-2 imagery were accessed from the [GEE platform (Sentinel-2 MSI, Level-2A)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR); and the annotations used in the dataset from the [Cropland Data Layer (CDL) by USDA NASS](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php), which were also accessed from the [GEE platform (USDA NASS Cropland Data Layer)](https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL)
