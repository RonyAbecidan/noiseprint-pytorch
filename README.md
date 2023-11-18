[![Generic badge](https://img.shields.io/badge/Library-Pytorch-<>.svg)](https://shields.io/) [![Ask Me Anything !](https://img.shields.io/badge/Official%20-No-1abc9c.svg)](https://GitHub.com/Naereen/ama) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=RonyAbecidan.noiseprint-pytorch)

Who has never met a forged picture on the web ? No one ! Everyday we are constantly facing fake pictures but it is not always easy to detect it.

In this repo, you will find an implementation of Noiseprint, a CNN-based camera model fingerprint. 

With this algorithm, you may find if an image has been falsified and even identify suspicious regions. A little example is displayed below.

![](https://i.imgur.com/7Y7a4YD.png)


It's a faifthful replica of the [official implementation](https://github.com/grip-unina/noiseprint/) using however the library Pytorch. To learn more about this network, I suggest you to read the paper that describes it [here](https://arxiv.org/pdf/1808.08396.pdf).

On top of Noiseprint, there is also several files containing pre-trained weights obtained by the authors which is compatible with this pytorch version.

Please note that the rest of the README is largely inspired by the original repo.

--- 

## Abstract

Forensic analyses of digital images rely heavily on the traces of in-camera and out-camera processes left on the acquired images. Such traces represent a sort of camera fingerprint. If one is able to recover them, by suppressing the high-level scene content and other disturbances, a number of forensic tasks can be easily accomplished. A notable example is the PRNU pattern, which can be regarded as a device fingerprint, and has received great attention in multimedia forensics. In this paper we propose a method to extract a camera model fingerprint, called noiseprint, where the scene content is largely suppressed and model-related artifacts are enhanced. This is obtained by means of a Siamese network, which is trained with pairs of image patches coming from the same (label +1) or different (label −1) cameras. Although noiseprints can be used for a large variety of forensic tasks, here we focus on image forgery localization. Experiments on several datasets widespread in the forensic community show noiseprint-based methods to provide state-of-the-art performance.

## License
Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document LICENSE.txt (included in this package)

## Where are the pre-trained weights coming from  ?

- Using images from a range of camera devices, the authors pre-trained their model across Quality Factors from 51 to 100, and they extended their training to encompass png images as well. As per the original repository's authors, the weights for the png images are refered to 'qf101' within the weight folder.

**_The pre-trained weights available in this repo are the results of these trainings achieved by the authors_**

**Remarks** : To train Noiseprint you need your own (relevant) datasets.

## Dependency
- **Pytorch** >= 1.8.1

## Demo
One may simply download the repo and play with the provided ipython notebook.

## N.B. :
- Considering that there is some differences between the implementation of common functions between Tensorflow/Keras and Pytorch, some particular methods of Pytorch (like batch normalization) are re-implemented here to match perfectly with the original Tensorflow version

- Noiseprint is an architecture difficult to train without GPU/Multi-CPU. Even in "eval" mode, if you want to use it for detecting forgeries in one image it may take some minutes using only your CPU. It depends on the size of your input image.


## Citation :

```js
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
} 
```
