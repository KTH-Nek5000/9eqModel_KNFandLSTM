# 9eqModel_LSTMandLKIS-DMD

## Introduction

The code in this repository features a Python implementation of the model for wall turbulence proposed by [Moehlis *et al.*](https://iopscience.iop.org/article/10.1088/1367-2630/6/1/056/meta) (2004, New J. Phys.). The time series generated are used to train neural networks that can predict the time evolution of the coefficients of the nine-equation model. More details about the implementation and the results from the training are available in ["Predictions of turbulent shear flows using deep neural networks", P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa](https://www.researchgate.net/publication/332495603_Predictions_of_turbulent_shear_flows_using_deep_neural_networks) (2019, *Phys. Rev. Fluids*)  

The LSTM predictions are compared with other Koopman-based frameworks such as LKIS-DMD

## Post-processing

### Lyapunov exponents

The script *Moehlis_perturb_generator.py* allows to generate timeseries that have a perturbation at time *t=500*. The script *Lyapunov_exponents.py* computes the Lyapunov exponents of these timeseries and it compares it with the LSTM predictions.
