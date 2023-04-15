# 2412_project
Exploring Pretrained Feature Extractors for Differentially Private Generative Modeling

## Experiment Results
[Google Drive Link](https://drive.google.com/file/d/1nR0wDjarNUtH99o7XbLVs1exvMhBTxh9/view?usp=share_link)


## Metric Implmentations
Uses [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch), and [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for metric evaluation. 

## Setup

Install pip packages in [requirements.txt](requirements.txt), download runs from Google Drive, unzip and move all ```runs_*``` folders from ```archive/``` into the local directory.

## File Organization
* ```data/``` holds all data.
* ```inception_score_pytorch/``` holds code from the implementation of [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch).
* ```output_plots``` stores all plot and visualization outputs.
* ```results``` stores CSVs of experimental metric evaluations.
* ```run_*``` holds experimental run folders, named by their args. Each experiment subfolder contains loss.txt, and checkpointed models and Opacus accountants.
* ```data.py``` handles data loading and partitioning.
* ```evaluate_metrics.py``` calculates IS and FID for a set of experiments.
* ```metrics.py``` makes the calls to calculate a single IS or FID metric.
* ```model_inversion.py``` performs all latent extraction using gradient ascent.
* ```models.py``` defines all models used in experiments.
* ```privacy.py``` calculates maximum gradient norm given a weight clip (from [DP-GAN](https://arxiv.org/abs/1802.06739) paper).
* ```report.pdf``` is the final report.
* ```testing.ipynb``` performs all the plotting and visualizations seen in the final report.
* ```train_autoencoder.py``` trains the public autoencoder.
* ```train_dp_vae.py``` trains the private VaeGM model in the latent space.
* ```train_dpgan.py``` implements and trains DP-GAN, WGAN, and WGAN-GP in the image space.
* ```train_latent_gan.py``` trains DP-GAN in the latent space.
* ```utils.py``` determines input hyperparameter args and generates and parses experiment folder names.

