import os
import kagglehub
from transformers import TimeSeriesTransformerConfig

SEED = 0

data_path = "./data/data.json"
model_path = "./model_checkpoint/checkpoint.pt"


# model configuration
time_step_interval = 5
context_length = 700
prediction_length = 50
lags_sequence = [3, 6, 12, 24, 49, 98, 147]

model_config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=2,
    num_time_features=6,
    num_dynamic_real_features=8,
    lags_sequence=lags_sequence,
    d_model=256,
    encoder_layers=2,
    decoder_layers=2,
    scaling="std",
    num_parallel_samples=32
)


# trading agent configuration
BALANCE = 1000
FEE = 0.47
MARGIN_RATIO = 0.175
ASSEST_RATIO = 0.8


# optimization
TOTAL_PROCESSES = 8
n_startup_trials = 160
n_trials = 1600


# backtest
backtest_optimized_algo = False # If false, use naive algorithm
optimized_algo_params = {
    "p_highs": 0.39,
    "p_lows": 0.66,
    "p_stoploss": 0.01,
    "using_dp": False
}

# where the output results from backtest and optimization go
results_dir = "./results" 