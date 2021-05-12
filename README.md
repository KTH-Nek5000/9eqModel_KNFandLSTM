# Recurrent neural networks and Koopman-based frameworks for temporal predictions in a low-order model of turbulence


## Introduction

The code in this repository features a Python implementation of recurrent neural networks and Koopman-based frameworks for prediction of temporal dynamics of a low-order model of near-wall turbulence proposed by [Moehlis *et al.*](https://iopscience.iop.org/article/10.1088/1367-2630/6/1/056/meta) (2004, New J. Phys.). The time series generated are used to train long-short-term memory (LSTM) networks and Koopman-based frameworks that can predict the time evolution of the coefficients of the nine-equation model. More details about the implementation and the results are available in ["Recurrent neural networks and Koopman-based frameworks for temporal predictions in a low-order model of turbulence", H. Eivazi, L. Guastoni, P. Schlatter, H. Azizpour, R. Vinuesa](https://arxiv.org/abs/2005.02762) (2021, *International Journal of Heat and Fluid Flow*)  

## Datasets

This folder contains the data files needed for training and testing of the models.

## HDMD

*HDMDs.py*: Python class for Hankel dynamic mode decomposition (HDMD).
*train_predict_HDMD.py*: train an HDMD model and uses it to predict time series based on a given input seed and reproduce the long-term statistics.

## KNF

*KNFs.py*: Python class for Koopman-based framework with nonlinear forcing (KNF).
*train_predict_KNF.py*: train a KNF model and uses it to predict time series based on a given input seed and reproduce the long-term statistics.

## LSTM

*LSTM1_t10000.h5*: a trained LSTM model. More details in ["Predictions of turbulent shear flows using deep neural networks", P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa](https://link.aps.org/doi/10.1103/PhysRevFluids.4.054603) (2019, *Phys. Rev. Fluids*; also available in [arXiv](https://arxiv.org/abs/1905.03634))

*predict_LSTM.py*: uses the trained LSTM model to predict time series based on a given input seed and reproduce the long-term statistics.
*lstm_pred_func.py*: Python function for the LSTM prediction iterations.


## Post-processing

*compare_short_term_predictions.py*: compares the performance of KNF, HDMD, and LSTM models in the short-term predictions by producing Figures 2 and 3 of the paper.
*compare_long_term_statistics.py*: compares the performance of KNF, HDMD, and LSTM models in the reproduction of the long-term statistics by producing Figures 5 of the paper.
*Moehlis_perturb_generator.py*: allows to generate time series that have a perturbation at time *t=500*. 
*Lyapunov_exponents.py*: computes the Lyapunov exponents of these time series and compares it with the LSTM predictions.

## Predictions

Results of the prediction of the time series and reproduction of the long-term statistics will be stored in this directory.

## Utilities

*optht.py*: Optimal hard threshold for singular values.
*statistics.py*: a Python function that computes the long-term statistics from given reference and predicted time series.
