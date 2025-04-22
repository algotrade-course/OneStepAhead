import numpy as np

from model import CustomTransformerModel
from utils import student_t_icdf

from typing import List, Tuple, Union


class TradingAgent():
    def __init__(
        self, 
        model, 
        balance:float = 1000, 
        fee: float = 0.47, 
        margin_ratio: float = 0.175,
        assest_ratio = 0.8,
        max_holdings: int = None
    ) -> None:
        self.model = model

        self.FEE = fee
        self.MARGIN_RATIO = margin_ratio
        self.ASSEST_RATIO = assest_ratio
        self.PRESERVATION = balance * (1 - assest_ratio)
        self.MINIMUM_BALANCE_TO_TRADE = balance * (1 - assest_ratio + 0.1)
        self.max_holdings = max_holdings

        self.holdings = []
        self.total_trade = 0
        self.win_cnt = 0
        self.balance = balance
        self.realized_asset = balance
        self.asset_history = [balance]

        self.pending_position = None

    
    def get_info(self) -> List[Union[int, float, List[float]]]:
        r"""
        Returns: total trade, win rate, asset history.
        """
        return [self.total_trade, self.win_cnt / (self.total_trade + 1e-9), self.asset_history]
    

    def open_position(self, position: dict) -> None:
        r"""
        Append a position to the list of holdings.
        """
        self.holdings.append(position)
        self.balance -= position["entry_point"] * self.MARGIN_RATIO

    
    def try_closing(self, curr_high: float, curr_low: float) -> None:
        r"""
        Close positions which hit take-profit or stop-loss.
        
        Args:
            curr_high: Latest High price.

            curr_low: Latest Low price.
        """
        total_realized_pnl = 0
        total_unrealized_pnl = 0
        new_holdings = []

        for position in self.holdings:
            isClosed = False
            realized_pnl = 0
            unrealized_pnl = 0

            # take-profit
            if (position["position_type"] == "LONG" and curr_high >= position["take_profit_point"]) or\
            (position["position_type"] == "SHORT" and curr_low <= position["take_profit_point"]):
                realized_pnl = abs(position["take_profit_point"] - position["entry_point"]) - self.FEE
            # stop-loss
            elif (position["position_type"] == "LONG" and curr_low <= position["stop_loss_point"]) or \
            (position["position_type"] == "SHORT" and curr_high >= position["stop_loss_point"]):
                realized_pnl = -abs(position["stop_loss_point"] - position["entry_point"]) - self.FEE
            else: # Check AR
                if position["position_type"] == "LONG":
                    unrealized_pnl = curr_high - position["entry_point"]
                else:
                    unrealized_pnl = position["entry_point"] - curr_low
                
                unrealized_balance = self.balance + total_unrealized_pnl + unrealized_pnl
                if unrealized_balance < self.PRESERVATION:
                    # force closing position
                    realized_pnl = unrealized_pnl - self.FEE
                    unrealized_pnl = 0


            if realized_pnl != 0:
                total_realized_pnl += realized_pnl
                self.balance += position["entry_point"] * self.MARGIN_RATIO + realized_pnl
                self.total_trade += 1
                if realized_pnl > 0:
                    self.win_cnt += 1
            else:
                new_holdings.append(position)
                total_unrealized_pnl += unrealized_pnl

        self.holdings = new_holdings
        self.realized_asset += total_realized_pnl 
        self.asset_history.append(self.realized_asset + total_unrealized_pnl)  
        # print(f"balance {self.balance:.4f} | total_unrealized_pnl {total_unrealized_pnl:.4f} | holdings {len(new_holdings)}")


    def close_all(self, close_price: float) -> None:
        r"""
        Force close all positions at close_price.
        
        Args:
            close_price:.
        """
        total_realized_pnl = 0

        for position in self.holdings:
            realized_pnl = 0
            if position["position_type"] == "LONG":
                realized_pnl = close_price - position["entry_point"] 
            else:
                realized_pnl = position["entry_point"] - close_price

            total_realized_pnl += realized_pnl
            self.balance += position["entry_point"] * self.MARGIN_RATIO + realized_pnl - self.FEE

            self.total_trade += 1
            if realized_pnl > 0:
                self.win_cnt += 1

        self.holdings = []
        self.realized_asset += total_realized_pnl
        self.asset_history.append(self.realized_asset)   


    def one_best_trade(
        self, 
        adjusted_prices: np.ndarray,
        adjusted_stoploss: np.ndarray,
        fee: float
    ) -> Union[List[float], List[None]]:
        """
        Long/Short based on max and min

        Returns: List[float]: [entry_point, take_profit_point, stop_loss_point].\n
            If there is no profitable trade, return list of None
        """
        maxima = np.argmax(adjusted_prices[0, :], axis=0)
        minima = np.argmin(adjusted_prices[1, :], axis=0)
        maximum = adjusted_prices[0, maxima]
        minimum = adjusted_prices[1, minima]

        entry_point = None
        take_profit_point = None
        stop_loss_point = None

        if maximum - minimum > fee:
            if maxima < minima:
                entry_point = maximum
                take_profit_point = minimum
                stop_loss_point = adjusted_stoploss[0, maxima]
            elif minima < maxima:
                entry_point = minimum
                take_profit_point = maximum
                stop_loss_point = adjusted_stoploss[1, minima]
        
        res = [entry_point, take_profit_point, stop_loss_point]
        res = [round(x, 1) if x is not None else None for x in res]
        return res
    

    def __call__(self, data: List, current_prices: List[float]) -> None:
        r"""
        Trade.

        Args:
            data: Processed data for prediction model.

            current_prices: Lastest High and Low price.
        """
        [curr_high, curr_low] = current_prices

        self.try_closing(curr_high, curr_low)
        
        if self.pending_position != None:
            # print(f"\n high {curr_high:.4f} | entry {self.pending_position['entry_point']:.4f} | low {curr_low:.4f}\n")
            # check if it is possible to enter pending position
            if (curr_high >= self.pending_position["entry_point"] and self.pending_position["entry_point"] >= curr_low):
                self.open_position(self.pending_position)
                # print("Open position")
                # self.pending_position = None
            # else:
                # print("Canceled")
            self.pending_position = None


        # If reached max_holdings, or having a pending position -> skip      
        if (self.max_holdings is not None and len(self.holdings) == self.max_holdings) or self.pending_position != None:
            return
        
        [
            past_values,
            past_additional_features, 
            past_masks, 
            future_additional_features
        ] = data
        
        # Move data to gpu
        device = self.model.device
        past_values = past_values.unsqueeze(0).to(device)
        past_additional_features = past_additional_features.unsqueeze(0).to(device)
        past_masks = past_masks.unsqueeze(0).to(device)
        future_additional_features = future_additional_features.unsqueeze(0).to(device)

        [means, stds] = self.model.generate(
            past_values=past_values, 
            past_time_features=past_additional_features, 
            past_observed_mask=past_masks, 
            future_time_features=future_additional_features
        ) # Predict future highs and lows


        # Values in the model was divided by ref price, so now we multiply it back
        means = means.cpu().numpy().squeeze(0).T

        [entry_point, take_profit_point, stop_loss_point] = self.one_best_trade(means, means, self.FEE)
        
        # check the balance and max_holdings
        if entry_point is not None and \
        entry_point * self.MARGIN_RATIO < self.MINIMUM_BALANCE_TO_TRADE:
            if entry_point < take_profit_point:
                pos_type = "LONG"
                stop_loss_point -= 0.1
            else:
                pos_type = "SHORT"
                stop_loss_point += 0.1

            self.pending_position = {
                "position_type": pos_type,
                "entry_point": entry_point,
                "take_profit_point": take_profit_point,
                "stop_loss_point": stop_loss_point
            }
    

class OptimizedTradingAgent(TradingAgent):
    r"""
    Adjusted Price using inverse cdf (quantile function).

    Use dynamic programming for trades selection.
    """
    def __init__(
        self, 
        model, 
        balance:float = 1000, 
        fee: float = 0.47, 
        margin_ratio: float = 0.175,
        assest_ratio = 0.8,
        p_value_of_highs: float = 0.5,
        p_value_of_lows: float = 0.5,
        p_diff_of_stoploss: float = 0.01,
        dynamic_programming: bool = False,
        max_holdings: int = None
    ) -> None:
        super().__init__(model, balance, fee, margin_ratio, assest_ratio, max_holdings)

        self.p_value_of_highs = p_value_of_highs
        self.p_value_of_lows = p_value_of_lows
        self.p_diff_of_stoploss = p_diff_of_stoploss
        self.dynamic_programming = dynamic_programming
    

    def best_trades(
        self, 
        adjusted_prices: np.ndarray,
        adjusted_stoploss: np.ndarray
    ) -> Union[List[float], List[None]]:
        """
        Find an optimal list trades, but only return the first one.

        Returns: List[float]: [entry_point, take_profit_point, stop_loss_point].\n
            If there is no profitable trade, return list of None
        """
        n = adjusted_prices.shape[-1]
        
        dp = [0] * (n + 1)
        total_risk = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(0, i - 1):
                [
                    entry_point, 
                    take_profit_point, 
                    stop_loss_point
                ] = self.one_best_trade(adjusted_prices[:, j+1 : i+1], adjusted_stoploss[:, j+1 : i+1], self.FEE)

                if entry_point is not None:
                    returns = abs(take_profit_point - entry_point) - self.FEE
                    risk =  abs(stop_loss_point - entry_point) + self.FEE

                    if dp[j] + returns > dp[i] or \
                    (  dp[j] + returns == dp[i] and total_risk[j] + risk < total_risk[i]):
                        dp[i] = dp[j] + returns
                        total_risk[i] = total_risk[j] + risk
        
        # Backtracking
        res = [None] * 3
        i = n
        while (dp[i] > 0):
            ok = False
            for j in range(0, i - 1):
                [
                    entry_point, 
                    take_profit_point, 
                    stop_loss_point
                ] = self.one_best_trade(adjusted_prices[:, j+1 : i+1], adjusted_stoploss[:, j+1 : i+1], self.FEE)

                if entry_point is not None:
                    returns = abs(take_profit_point - entry_point) - self.FEE
                    risk =  abs(stop_loss_point - entry_point) + self.FEE

                    if dp[j] + returns == dp[i] and total_risk[j] + risk == total_risk[i]:
                        res = [entry_point, take_profit_point, stop_loss_point]
                        i = j
                        ok = True
                        break;
            if not ok:
                i -= 1

        return res


    def __call__(
        self, 
        data: List, 
        current_prices: List[float], 
        adjusted_prices: np.ndarray = None,
        adjusted_stoploss: np.ndarray = None
    ) -> None:
        r"""
        Trade.

        Args:
            data: Processed data for prediction model.

            current_prices: Lastest High and Low price.

            adjusted_prices: pre-computed, use when searching parameters in optimization phase.
        """
        [curr_high, curr_low] = current_prices

        self.try_closing(curr_high, curr_low)
        
        # try opening
        if self.pending_position != None:
            # print(f"\n high {curr_high:.4f} | entry {self.pending_position['entry_point']:.4f} | low {curr_low:.4f}\n")
            # check if it is possible to enter pending position
            if (curr_high >= self.pending_position["entry_point"] and self.pending_position["entry_point"] >= curr_low):
                self.open_position(self.pending_position)
                # print("\nOpen position")
                # profit = abs(self.pending_position['entry_point'] - self.pending_position['take_profit_point'])
                # loss = abs(self.pending_position['entry_point'] - self.pending_position['stop_loss_point'])
                # print(f"Return/Risk: {profit}/{loss}")
                # self.pending_position = None
            # else:
                # print("Canceled")
            self.pending_position = None


        # If reached max_holdings, or having a pending position -> skip
        if (self.max_holdings is not None and len(self.holdings) == self.max_holdings) or self.pending_position != None:
            return
        

        [
            past_values,
            past_additional_features, 
            past_masks, 
            future_additional_features
        ] = data

        if adjusted_prices is None: 
            # Move data to device
            device = self.model.device
            past_values = past_values.unsqueeze(0).to(device)
            past_additional_features = past_additional_features.unsqueeze(0).to(device)
            past_masks = past_masks.unsqueeze(0).to(device)
            future_additional_features = future_additional_features.unsqueeze(0).to(device)

            # Predict future highs and lows
            [means, stds] = self.model.generate(
                past_values=past_values, 
                past_time_features=past_additional_features, 
                past_observed_mask=past_masks, 
                future_time_features=future_additional_features
            )

            means = means.cpu().numpy().squeeze(0).T
            stds = stds.cpu().numpy().squeeze(0).T

            # adjust prices
            adjusted_prices = np.zeros_like(means)
            adjusted_prices[0, :] = student_t_icdf(self.p_value_of_highs, means[0, :], stds[0, :]) # Highs
            adjusted_prices[1, :] = student_t_icdf(self.p_value_of_lows, means[1, :], stds[1, :]) # Lows

            # print(f"highs: max {np.max(adjusted_prices[0, :])} | min {np.min(adjusted_prices[0, :])}")
            # print(f"lows: max {np.max(adjusted_prices[1, :])} | min {np.min(adjusted_prices[1, :])}")

            adjusted_stoploss = np.zeros_like(means)
            adjusted_stoploss[0, :] = student_t_icdf(self.p_value_of_highs + self.p_diff_of_stoploss, means[0, :], stds[0, :]) # Highs
            adjusted_stoploss[1, :] = student_t_icdf(self.p_value_of_lows - self.p_diff_of_stoploss, means[1, :], stds[1, :]) # Lows

        else:
            adjusted_prices = np.array(adjusted_prices) # copy
            adjusted_stoploss = np.array(adjusted_stoploss)


        # make decision
        if self.dynamic_programming:
            [entry_point, take_profit_point, stop_loss_point] = self.best_trades(adjusted_prices, adjusted_stoploss)
        else:
            [entry_point, take_profit_point, stop_loss_point] = self.one_best_trade(adjusted_prices, adjusted_stoploss, self.FEE)

        # check balance
        if entry_point is not None and \
        entry_point * self.MARGIN_RATIO < self.MINIMUM_BALANCE_TO_TRADE:
            pos_type = "LONG" if entry_point < take_profit_point else "SHORT"
            self.pending_position = {
                "position_type": pos_type,
                "entry_point": entry_point,
                "take_profit_point": take_profit_point,
                "stop_loss_point": stop_loss_point
            }
        