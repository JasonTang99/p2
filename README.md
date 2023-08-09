# Exploring Pretrained Feature Extractors for Differentially Private Generative Modeling

For a more detailed description of the project, please see the [final report](report/report.pdf).

## Summary
This project explores the use of autoencoders and Variational Autoencoders (VAEs) as alternatives to Generative Adversarial Networks (GANs) in the [Differential Privacy for Model Inversion (DPMI)](https://arxiv.org/abs/2201.03139) framework presented by Chen et al. The aim is to generate synthetic data for privacy-preserving applications with improved stability and simplicity compared to GANs. We conduted comprehensive experiments that assess the performance of these models across stability, privacy, and utility aspects, employing metrics like Inception Score (IS) and Fréchet Inception Distance (FID). The findings suggest that autoencoders and VAEs present promising alternatives to GANs, particularly in scenarios where training GANs poses challenges.

<!-- Image of report/images/setup.png -->
![Alt text](report/images/setup.png?raw=true "Setup")

## Experiments
We utilize [PyTorch](https://pytorch.org/) and [Opacus](https://opacus.ai/) for implementation and privacy accounting, and the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) to explore multiple variations of the DPMI framework. The experiments compare autoencoders, VAEs, and GANs with a focus on stability, privacy, and utility metrics. The study also delves into the influence of different latent extraction techniques, activation functions, and gradient norm clipping values.

<!-- Generated Data -->
#### Generations at different epsilons
![Alt text](report/images/generated_data.png?raw=true "Generated Data")
<!-- public_gan_training -->
#### GAN Mode Collapse
![Alt text](report/images/public_gan_training.png?raw=true "Public GAN Training")
<!-- during_train_inf -->
#### Latent Extraction
![Alt text](report/images/during_train_inf.png?raw=true "Latent Extraction")

## Results
The project's results underscore the potential viability of autoencoders and VAEs as substitutes for GANs within the DPMI framework. These alternatives exhibit simplicity and stability in generating synthetic data for privacy-preserving applications, particularly when faced with inadequate data or resource availability that poses challenges for training GANs in conventional DPMI scenarios.

<!-- activation -->
#### Activation Functions
![Alt text](report/images/activation.png?raw=true "Activation Functions")
<!-- c_p -->
#### Gradient Norm Clipping
![Alt text](report/images/c_p.png?raw=true "Gradient Norm Clipping")

<!-- #################################################### -->
## Metric Implmentations
Uses [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch), and [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for metric evaluation. 

## Setup
Experiment Results: [Google Drive Link](https://drive.google.com/file/d/1esmQm1gU3tqJppAE56oRRpXia4CMu4Sc/view?usp=drive_link)

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

