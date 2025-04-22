import os
import json
import torch
import random
import numpy as np
import scipy.stats as st

from matplotlib import pyplot as plt

from typing import List, Tuple


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maximum_drawdown(asset_history: List[float]) -> float:
    peak = -1e9
    mdd = 0
    for x in asset_history:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1)
    return mdd * 100


def period_return(asset_history: List[float], period: int) -> List[float]:
    returns = []
    n = len(asset_history)
    for i in range(0, n, period):
        returns.append(asset_history[min(n - 1, i + period)] / (asset_history[i] + 1e-9) - 1)
    return returns

def sharpe_ratio(daily_returns, risk_free_rate=0.03, trading_days_per_year=252):
    """
    Calculate the Sharpe Ratio.
    
    Parameters:
    - daily_returns: list or array of daily returns (as decimals, e.g., 0.01 for 1%)
    - risk_free_rate: annual risk-free rate (as decimal, e.g., 0.03 for 3%)
    - trading_days_per_year: number of trading days in a year (default: 252)
    
    Returns:
    - Annualized Sharpe Ratio
    """
    daily_returns = np.array(daily_returns)
    excess_daily_returns = daily_returns - (risk_free_rate / trading_days_per_year)
    mean_excess_return = np.mean(excess_daily_returns)
    std_dev = np.std(daily_returns, ddof=1)
    
    sharpe_ratio = (mean_excess_return / std_dev) * np.sqrt(trading_days_per_year)
    return sharpe_ratio


def student_t_icdf(
    p: float, 
    means: torch.Tensor | np.ndarray, 
    stds: torch.Tensor | np.ndarray, 
    df: float = 1
) -> torch.Tensor | np.ndarray:
    """
    Compute the inverse CDF (quantile) for a Student's t distribution with
    specified mean, standard deviation, and degrees of freedom.

    Args:
      p (float): Lower tail probability (values in (0,1))
      mean (torch.Tensor): The location parameters (mean)
      std (torch.Tensor): The scale parameters (standard deviation)
      df (float): Degrees of freedom of the Student's t distribution

    Returns:
      torch.Tensor: The quantiles corresponding to probability p
    """
    quantile = st.t.ppf([p], df)[0]
    
    return means + stds * quantile


class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def output_results(results: Tuple[int, float, List[float]], dir: str = "./"):
    """
    Args:
        results: (total_trades, win_rate, asset_history)
    """
    os.makedirs(dir, exist_ok=True)

    (total_trades, win_rate, asset_history) = results
    print(f"Trades: {total_trades}")
    print(f"Win rate: {win_rate * 100:.4f}%")
    print(f"Assest: {asset_history[-1]}")

    # write results to json file
    res_dict = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "asset_history": asset_history
    }
    with open(os.path.join(dir, "results.json"), "w") as fout:
        json.dump(res_dict, fout, cls=NumpyTypeEncoder)


    accum_return_rate = (asset_history[-1] / asset_history[0] - 1) * 100
    print(f"accum_return_rate: {accum_return_rate:.4f}%")

    mdd = maximum_drawdown(asset_history)
    print(f"maximum drawdown: {mdd:.4f}%")

    # plot daily return
    daily_returns = period_return(asset_history, period=49)
    x = list(range(len(daily_returns)))
    plt.plot(x, daily_returns)
    plt.title(label="Daily Return")
    plt.savefig(os.path.join(dir, "daily_return.png"))
    plt.show()
    plt.clf()

    # plot asset history
    x = list(range(len(asset_history)))
    plt.plot(x, asset_history)
    plt.title(label="Asset history")
    plt.savefig(os.path.join(dir, "asset_history.png"))
    plt.show()
    plt.clf()

    # annualized Sharpe ratio
    sharpe = sharpe_ratio(daily_returns, risk_free_rate=0.03, trading_days_per_year=252)
    print(f"sharpe: {sharpe:.4f}")