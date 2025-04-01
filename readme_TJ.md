Please acknowledge the use of these scripts in any publications which make use of them.

I constructed Reservoir (without any ``Class" setting!) and realtime filter (based on Bayesian optimization) code.

If you want to know the mathematical structure of this realtime filter, see the following paper:

``Long-term prediction of El Niño-Southern Oscillation using reservoir computing with data-driven realtime filter"

https://arxiv.org/abs/2501.17781

# RealtimeFilter_TsuyoshiYoneda.py

Optimizing the hyperparameters of the realtime-filter.
The sample time series stored in "lorentz_long_x.csv" is the x-component of the Lorenz 63 dynamical system model.

You can set desirable ranges of the hyperparameters
- r_1: Higher end of the passing frequency range
- r_2: Lower end of the passing frequency range
- c: Parameter for the attenuation of weight function
- d_1: Amplitude of the frequency r_1
- d_2: Amplitude of the frequency r_2
- width: Length of weighted running average in past time series

Make sure that the parameter search space is identical between the "frozen trial" section and the "objective function" section.

The following parameters are related to the reduction of the indefinite factor of the time series.
- T_train: Training length
- n_trials: Number of trials for optuna
- max_dim: Max pattern length
- min_dim: Min pattern length
- min_count: Delete patterns if occurrence is fewer than this
- n_bins: Number of partitions of range
- min_win_vote_rate: Minimum requirement for win_vote_rate

The history of bayesian optimization, "optuna_filter_params.csv", and the filtered time series "data_flt.csv" are generated.

# Reservoir_TsuyoshiYoneda.py

Optimizing the hyperparameters of the echo state network model.
You can set desirable ranges of the hyperparameters
- lag: Delay time 
- dim: Dimension of input delay-coordinate vector
- N_x: Dimension of reservoir state vector
- beta: Regularization parameter to avoid over-fitting
- density: Rate of non-zero elements in recurrent connection matrix A
- input_scale: Scaling parameter of the input matrix
- rho: maximum Eigenvalue of the matrix A
- alpha: Leaking rate
- seed_value: Seed value to generate random matrices

Make sure that the parameter search space is identical between the "frozen trial" section and the "objective function" section.

The following parameters are related to training and prediction of the time series.
- n_trials: Number of times to test with optuna (generate autocorrelation for this number of times)
- T_train: Training period
- T_test: Forecast period
- test_num: Number of times for one test (to get one autocorrelation)
- discard_len: Number of steps not learned at the beginning (not important)
Note that the number of reservoir node is fixed to 70 in the sample code.

The history of bayesian optimization, "reservoir.csv" is generated.