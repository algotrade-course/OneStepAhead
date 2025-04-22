import torch
import numpy as np

from torch.utils.data import Dataset

from typing import List

def split_data(data, ratio):
    """
    return train, val, test splitted from data according to the given ratio.
    """
    n = len(data)
    s = sum(ratio)

    train_cnt = int(ratio[0] / s * n)
    val_cnt = int(ratio[1] / s * n)
    
    return data[: train_cnt], \
            data[train_cnt : train_cnt + val_cnt], \
            data[train_cnt + val_cnt :]


def merge_2_records(last_record, new_record):
    record = last_record.copy()
    if last_record["Mask"] == 0:
        record = new_record
    elif new_record["Mask"] == 1:
        record["High"] = max( float(record["High"]), float(new_record["High"]) )
        record["Low"] = min( float(record["Low"]), float(new_record["Low"]) )
        record["Close"] = float(new_record["Close"])
        record["Volume"] += int(new_record["Volume"])
        record["Mask"] = 1
    return record


def convert_records_list(data, n):
    r"""
    Convert 1-minute record list into n-minute record list.
    """
    new_data = []
    current_day = None
    current_vn30 = None
    current_vn30f = None
    cnt = 0
    for record in data:
        if record["Time"]["DayOfYear"] == current_day and cnt < n:
            cnt += 1
            current_vn30f = merge_2_records(current_vn30f, record["VN30F"])
            current_vn30 = merge_2_records(current_vn30, record["VN30"])
        else:
            if cnt > 0:
                time = record["Time"]
                time["TimeOfDay"] = time["TimeOfDay"] // n # merge records -> the index of a record in a day changes
                new_data.append({
                    "Time": time,
                    "VN30F": current_vn30f,
                    "VN30": current_vn30
                })

            cnt = 1
            current_vn30f = record["VN30F"]
            current_vn30 = record["VN30"]
            current_day = record["Time"]["DayOfYear"]

    if cnt > 0:
        time = record["Time"]
        time["TimeOfDay"] = time["TimeOfDay"] // n
        new_data.append({
            "Time": time,
            "VN30F": current_vn30f,
            "VN30": current_vn30
        })
    
    return new_data


def cyclic_encoding(value, period):
    return np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)


def encode_time_feature(data):
    r"""
    Encode the time features. The structure of data does not change.
    """
    day_of_year_period = 366
    expiration_period = 40
    time_of_day_period = 0

    cnt = 0
    current_day_of_year = -1
    for record in data:
        if record["Time"]["DayOfYear"] != current_day_of_year:
            time_of_day_period = max(time_of_day_period, cnt)
            current_day_of_year = record["Time"]["DayOfYear"]
            cnt = 1
        else:
            cnt += 1

    time_of_day_period = max(time_of_day_period, cnt)
    time_of_day_period *= 2 # I dont know. Maybe for futureproof, in case the trading hours change
    
    for record in data:        
        encoded_expiration = cyclic_encoding(record["Time"]["Expiration"], expiration_period)
        encoded_time_of_day = cyclic_encoding(record["Time"]["TimeOfDay"], time_of_day_period)
        encoded_day_of_year = cyclic_encoding(record["Time"]["DayOfYear"], day_of_year_period)

        record["Time"] = {
            "Expiration_sin": encoded_expiration[0],
            "Expiration_cos": encoded_expiration[1],
            "TimeOfDay_sin": encoded_time_of_day[0],
            "TimeOfDay_cos": encoded_time_of_day[1],
            "DayOfYear_sin": encoded_day_of_year[0],
            "DayOfYear_cos": encoded_day_of_year[1]
        }
    return data


def to_tensor(data):
    """
        target_features (n, 2):
            HL_VN30F

        additional_features (n, 14):
            Expiration_sin,
            Expiration_cos,
            TimeOfDay_sin,
            TimeOfDay_cos
            DayOfYear_sin,
            DayOfYear_cos,
            OCV_VN30F,
            OHLCV_VN30
            
        masks (n, 1)
    """
    target_keys = ["High", "Low"]
    target_features = torch.Tensor([[x["VN30F"][key] if x["VN30F"]["Mask"] else 0 for key in target_keys] for x in data])

    vn30f_keys = ["Open", "Close", "Volume"]
    vn30_keys = ["Open", "High", "Low", "Close", "Volume"]
    additional_features = torch.cat(
        (
            torch.Tensor([list(x["Time"].values()) for x in data]), # 6 time features,
            torch.Tensor([[x["VN30F"][key] if x["VN30F"]["Mask"] else 0 for key in vn30f_keys] for x in data]), # 3 vn30f
            torch.Tensor([[x["VN30"][key] if x["VN30"]["Mask"] else 0 for key in vn30_keys] for x in data]) # 5 vn30
        ),
        dim=-1
    )

    target_masks = torch.Tensor([[x["VN30F"]["Mask"]] for x in data])

    # print(target_features.shape)
    # print(additional_features.shape)
    # print(target_masks.shape)
    # print("==================")

    return target_features, additional_features, target_masks


def standard_scale_positives(input):
    """
    Apply standard scalling to positive elements of a tensor.
    """
    values = input[input > 0]
    n = len(values)
    mean = torch.mean(values)
    smooth = 1e-9
    std = torch.sqrt( torch.sum((values - mean)**2) / max(smooth, (n - 1)) )
    result = torch.where(input > 0, (input - mean) / (std + smooth), 0)
    return result


def normalize_data(values, additional_features, ref_price):
    # Take a look at to_tensor() for more information about the values in these tensor

    additional_features[:, 8] = standard_scale_positives(additional_features[:, 8]) # vn30f volumes
    additional_features[:, 13] = standard_scale_positives(additional_features[:, 13]) # vn30 volumes

    # Scale ohlc based on vn30f reference price
    values /= ref_price 
    additional_features[:, 6 : 8] /= ref_price
    additional_features[:, 9 : -1] /= ref_price
    
    return values, additional_features



class StockPriceDateset(Dataset):
    def __init__(self, data, token_interval = 5, context_length = 250, prediction_length = 50, normalization: bool = False):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalization = normalization

        data = convert_records_list(data, token_interval) # n-minutes candlesticks
        data = encode_time_feature(data) # encode cyclic data with sine and cos

        self.ref_prices = [x["VN30F"]["RefPrice"] for x in data]

        self._size = len(data) - context_length - prediction_length + 1

        self.target_features, self.additional_features, self.target_masks = to_tensor(data)


    def __len__(self):
        return self._size
    
    def __iter__(self):
        for i in range(self._size):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        if index < 0:
            index += self._size
            
        past_begin = index
        past_end = past_begin + self.context_length
        future_begin = past_end
        future_end = future_begin + self.prediction_length

        # Get data and apply normalization
        values = self.target_features[past_begin : future_end].clone()
        additional_features = self.additional_features[past_begin : future_end].clone()
        if self.normalization:
            values, additional_features = normalize_data(values, additional_features, self.ref_prices[past_end - 1])

        past_values = values[: self.context_length]
        future_values = values[self.context_length :]

        past_additional_features = additional_features[: self.context_length]

        # Dummy future_additional_features
        sz = list(past_additional_features.size())
        sz[0] = self.prediction_length
        future_additional_features = torch.zeros(sz)

        # future_additional_features = additional_features[self.context_length :]
        # if (future_additional_features.shape[0] < self.prediction_length):
        #     print("dataset size:", self._size)
        #     print("index:", index)
        #     print("item length:", future_additional_features.shape)

        past_masks = torch.ones(self.context_length).unsqueeze(-1)
        future_masks = self.target_masks[future_begin : future_end]

        return [past_values, past_additional_features, past_masks, future_values, future_additional_features, future_masks, self.ref_prices[past_end - 1]]
    

# if __name__ == "__main__":
#     data_path = "/home/mtdat/code/CS408-ComputationalFinance/code/data.json"
#     with open(data_path, "r") as fin:
#         data = json.load(fin)

#     dataset = StockPriceDateset(data)

#     print(f"Dataset size: {len(dataset)}")

#     indices = np.random.choice(dataset.size(), 5)

#     for i in indices:
#         (past, past_mask, past_time_features, future, future_masks, future_time_features) = dataset[i]
#         print(past.shape)
#         print(past_mask.shape)
#         print(past_time_features.shape)
#         print(future.shape)
#         print(future_masks.shape)
#         print(future_time_features.shape)
#         break