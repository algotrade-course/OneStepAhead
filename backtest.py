import os
import json
import copy
import torch
import numpy as np

from configs import config
from utils import set_seed, output_results
from dataset import StockPriceDateset, split_data
from model import CustomTransformerModel, load_checkpoint
from trading_agent import TradingAgent, OptimizedTradingAgent


def backtest(
    trader: TradingAgent, 
    dataset: StockPriceDateset, 
    adjusted_prices_list: list[np.ndarray] = None, 
    adjusted_stoploss_list: list[np.ndarray] = None, 
    verbose: bool = False
) -> tuple[int, float, list[float]]:
    """
    Args:
        adjusted_prices_list: pre-computed model's prediction.\
        Use when running backtest multiple time in optimization.
    """
    
    for i, (past_values, past_additional_features, past_masks, feature_values, future_additional_features, _, ref_price) in enumerate(dataset):
        data = [
            past_values,
            past_additional_features,
            past_masks,
            future_additional_features
        ]
        current_prices = past_values[-1].tolist()
        # print("current_prices", current_prices)

        if min(current_prices) == 0:
            continue

        # print("\n=========================")
        # print(f"feature_values: max {torch.max(feature_values)} | min {torch.min(feature_values)}")
        if adjusted_prices_list is not None:
            trader(data, current_prices, adjusted_prices_list[i], adjusted_stoploss_list[i])
        else:
            trader(data, current_prices)
        
        total_trades, win_rate, asset_history = trader.get_info()
        if verbose:
            print(f"Testing {i + 1}/{len(dataset)}: trades {total_trades}, win rate {win_rate:.4f}, assest {asset_history[-1]:.4f}", end='\r')

    # (last window, past_additional_features, last time step, vn30f close price)
    close_price = dataset[-1][1][-1][7].item()
    trader.close_all(close_price)

    total_trades, win_rate, asset_history = trader.get_info()
    if verbose:
        print(f"Testing Complete: trades {total_trades}, win rate {win_rate:.4f}, assest {asset_history[-1]:.4f}" + " " * 10)

    return [total_trades, win_rate, asset_history]


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

    # Model
    model = CustomTransformerModel(config.model_config)
    load_checkpoint(config.model_path, model)
    model = model.eval()
    model = model.to(device)

    trader0 = None
    if config.backtest_optimized_algo:
        trader0 = OptimizedTradingAgent(
            model, 
            p_value_of_highs=config.optimized_algo_params["p_highs"], 
            p_value_of_lows=config.optimized_algo_params["p_lows"], 
            p_diff_of_stoploss=config.optimized_algo_params["p_stoploss"], 
            dynamic_programming=config.optimized_algo_params["using_dp"]
        )
    else:
        trader0 = TradingAgent(model)


    set_seed(config.SEED)
    print("In sample backtesting:")
    trader = copy.deepcopy(trader0)
    backtest_results = backtest(trader, in_sample_set, verbose=True)
    output_results(backtest_results, os.path.join(config.results_dir, "in_sample_backtesting"))


    set_seed(config.SEED)
    print("Out of sample backtesting:")
    trader = copy.deepcopy(trader0)
    backtest_results = backtest(trader, out_of_sample_set, verbose=True)
    output_results(backtest_results, os.path.join(config.results_dir, "out_of_sample_backtesting"))