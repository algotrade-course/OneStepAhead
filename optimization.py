import os
import copy
import json
import torch
import optuna
import numpy as np

from joblib import Memory
from multiprocessing import Process
from torch.utils.data import DataLoader
from optuna.visualization import plot_optimization_history

from configs import config
from utils import set_seed, student_t_icdf, output_results, period_return, sharpe_ratio
from dataset import StockPriceDateset, split_data
from model import CustomTransformerModel, load_checkpoint
from trading_agent import OptimizedTradingAgent
from backtest import backtest


STORAGE_FILE = "optuna_study.db"
STORAGE_PATH = f"sqlite:///{STORAGE_FILE}"


# cache 
memory = Memory("./cachedir", verbose=0)
@memory.cache(ignore=["model", "dataset"])
def predict(model, dataset: StockPriceDateset) -> tuple[torch.Tensor, torch.Tensor]:
    means_list = []
    stds_list = []
    dfs_list = []

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    for i, (past_values, past_additional_features, past_masks, _, future_additional_features, _, _) in enumerate(dataloader):
        device = model.device
        past_values = past_values.to(device)
        past_additional_features = past_additional_features.to(device)
        past_masks = past_masks.to(device)
        future_additional_features = future_additional_features.to(device)

        means, stds, dfs = model.generate(
            past_values=past_values, 
            past_time_features=past_additional_features, 
            past_observed_mask=past_masks, 
            future_time_features=future_additional_features
        ) # (batch_size, 50, 2)

        means_list.append(means)
        stds_list.append(stds)
        dfs_list.append(dfs)
        print(f"Predicting: {i + 1}/{len(dataloader)}", end='\r')
    print()

    # (dataset_len, 50, 2)
    means = torch.cat(means_list).permute((0, 2, 1)).cpu().numpy()
    stds = torch.cat(stds_list).permute((0, 2, 1)).cpu().numpy()
    dfs = torch.cat(stds_list).permute((0, 2, 1)).cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stds, means, dfs

@memory.cache
def get_adjusted_highs(p):
    return student_t_icdf(p, means[:, 0, :], stds[:, 0, :], dfs[:, 0, :])

@memory.cache
def get_adjusted_lows(p):
    return student_t_icdf(p, means[:, 1, :], stds[:, 1, :], dfs[:, 1, :])


def get_adjusted_prices_and_stoploss(p_highs: float, p_lows: float, p_stoploss: float) -> tuple[np.ndarray, np.ndarray]:
    p_stoploss_highs = p_highs + p_stoploss
    p_stoploss_lows = p_lows - p_stoploss

    adjusted_highs = get_adjusted_highs(p_highs)
    adjusted_lows = get_adjusted_lows(p_lows)

    adjusted_stoploss_highs = get_adjusted_highs(p_stoploss_highs)
    adjusted_stoploss_lows = get_adjusted_lows(p_stoploss_lows)

    adjusted_prices = np.stack((adjusted_highs, adjusted_lows), axis=1)
    adjusted_stoploss = np.stack((adjusted_stoploss_highs, adjusted_stoploss_lows), axis=1)

    return (adjusted_prices, adjusted_stoploss)


def backtest_wrapper(p_highs: float, p_lows: float, p_stoploss: float, using_dp: bool):
    (adjusted_prices, adjusted_stoploss) = get_adjusted_prices_and_stoploss(p_highs, p_lows, p_stoploss)

    trader = OptimizedTradingAgent(
        None, 
        p_value_of_highs=p_highs, 
        p_value_of_lows=p_lows,
        p_diff_of_stoploss=p_stoploss,
        dynamic_programming=using_dp,
        balance=config.BALANCE,
        fee=config.FEE,
        margin_ratio=config.MARGIN_RATIO,
        assest_ratio=config.ASSEST_RATIO
    )

    [_, _, asset_history] = backtest(trader, dataset, adjusted_prices, adjusted_stoploss)

    daily_returns = period_return(asset_history, period=49)
    sharpe = sharpe_ratio(daily_returns, risk_free_rate=0.03, trading_days_per_year=252)

    return sharpe


def objective(trial):
    set_seed(config.SEED + trial.number)

    p_highs = trial.suggest_float("p_highs", 0.10, 0.90, step=0.01)
    p_lows = trial.suggest_float("p_lows", 0.10, 0.90, step=0.01)

    max_p_stoploss = min(p_lows - 0.01, 0.99 - p_highs)
    p_stoploss = trial.suggest_float("p_stoploss", 0.01, max_p_stoploss, step=0.01)
    using_dp = trial.suggest_int("using_dp", 0, 1, step=1)
    
    result = backtest_wrapper(p_highs, p_lows, p_stoploss, using_dp)
    return result


def run_study(offset, startup_trials_per_process, trials_per_process):
    sampler = optuna.samplers.TPESampler(
        seed=config.SEED + offset, 
        n_startup_trials=startup_trials_per_process, 
        multivariate=True,
        warn_independent_sampling=False
    )
    study = optuna.load_study(study_name="parallel_reproducible_study", storage=STORAGE_PATH, sampler=sampler)
    study.optimize(objective, n_trials=trials_per_process, n_jobs=1)

    return study.best_params


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    set_seed(config.SEED)

    # Data Preparation
    with open(config.data_path, "r") as fin:
        data = json.load(fin)

    _, in_sample_data, out_of_sample_data = split_data(data, ratio=(6, 2, 2))
    max_lag = max(config.lags_sequence)
    in_sample_set = StockPriceDateset(
        data=in_sample_data,
        token_interval=config.time_step_interval,
        context_length=config.context_length + max_lag,
        prediction_length=config.prediction_length
    )
    out_of_sample_set = StockPriceDateset(
        data=out_of_sample_data,
        token_interval=config.time_step_interval,
        context_length=config.context_length + max_lag,
        prediction_length=config.prediction_length
    )


    model = CustomTransformerModel(config.model_config)
    load_checkpoint(config.model_path, model)
    model = model.eval()
    model = model.to(device)

    dataset = in_sample_set
    # (dataset_len, 2, 50)
    means, stds, dfs = predict(model, dataset)


    # create optuna study
    startup_trials_per_process = config.n_trials / config.TOTAL_PROCESSES
    trials_per_process = config.n_trials / config.TOTAL_PROCESSES
    study = optuna.create_study(
        study_name="parallel_reproducible_study",
        direction="maximize",
        storage=STORAGE_PATH
    )


    # optimize
    processes = []
    for i in range(config.TOTAL_PROCESSES):
        p = Process(target=run_study, args=(i, startup_trials_per_process, trials_per_process))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    study = optuna.load_study(study_name="parallel_reproducible_study", storage=STORAGE_PATH)

    filtered_trials = [t for t in study.trials if t.value is not None and t.value > -10]

    # Create a new in-memory study to store the filtered trials
    filtered_study = optuna.create_study(direction=study.direction)
    for trial in filtered_trials:
        filtered_study.add_trial(trial)
    # plot_optimization_history(study, target_name="Sharpe Ratio").show()

    fig = plot_optimization_history(study, target_name="Sharpe Ratio")
    fig.write_image(os.path.join(config.results_dir, "optimization_history.png"), width=1200, height=675)

    # output best params
    best_params = study.best_trial.params
    print(f"Best trial: {study.best_trial.value}, params: {best_params}")

    if config.results_dir != "":
        os.makedirs(config.results_dir, exist_ok=True)

    params_path = os.path.join(config.results_dir, "best_params.json")
    with open(params_path, "w") as fout:
        json.dump(best_params, fout)


    # run backtests
    trader0 = OptimizedTradingAgent(
        model, 
        p_value_of_highs=best_params["p_highs"], 
        p_value_of_lows=best_params["p_lows"], 
        p_diff_of_stoploss=best_params["p_stoploss"],
        dynamic_programming=best_params["using_dp"],
        balance=config.BALANCE,
        fee=config.FEE,
        margin_ratio=config.MARGIN_RATIO,
        assest_ratio=config.ASSEST_RATIO
    )

    set_seed(config.SEED)
    print("In sample backtesting:")
    trader = copy.deepcopy(trader0)
    backtest_results = backtest(trader, in_sample_set, verbose=True)
    output_results(backtest_results, os.path.join(config.results_dir, "in_sample_optim"))


    set_seed(config.SEED)
    print("Out of sample backtesting:")
    trader = copy.deepcopy(trader0)
    backtest_results = backtest(trader, out_of_sample_set, verbose=True)
    output_results(backtest_results, os.path.join(config.results_dir, "out_of_sample_optim"))