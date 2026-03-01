# Alzheimer’s Detection
<!--
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](#)<br>
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Jeetu95/Brain-Tumor-Segmentation/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/Jeetu95/Brain-Tumor-Segmentation/issues)
-->
![Alzheimer’s Detection](./readme/first_look.gif)


## Table of Contents


>- [Introduction](#Introduction)
>- [Training](#Training)
>- [Results](#Results)
>- [Installation](#Installation)
>- [How to Use](#How_to_Use)


### Introduction:

We propose to implement a deep-learning framework that will segment the affected part of the brain in the MRI scans along with predicting severity of the alzheimer's disease.

We are going to deploy the model with simple interactive user interface - **Alzheimer’s Detection** on web platform.

The [demo_video](./demo/site_demo-2020-12-19_11.03.47.mp4)

### Training

- For Classification:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x57iiqQiPfG6mSG2dKtLMQkp1CaNDchy?usp=sharing#scrollTo=cKb2jNWrdTU9)

- For Semantic Segmentation:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u-vSPhU_hwaZdwsuABK4xCRet_jEp0Tu?usp=sharing)

### Results
Final Mean Dice  Loss - 0.74461 for segmentation
ROC AUC score - 0.946 for classification
### Installation

Fork this repository.

Then move to the desired directory in your PC and type in the terminal

```shell
git clone https://github.com/chetanpandey1266/FallFest-VWXYZ.git

cd web_app_with_ml_backend

pip install -r requirements.txt
```

Now your webapp is ready to run on your localhost😄

To run the web app

```shell

python app.py
```

This will run your site on localhost

### How_to_Use


For demo refer this [video](./demo/site_demo-2020-12-19_11.03.47.mp4)



