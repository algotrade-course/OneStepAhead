import os
import time
import json
from datetime import datetime, date, timedelta
from ssi_fc_data import fc_md_client, model

from configs import config, fc_data_config


def get_third_thursday(year: int, month: int):
    day = 1
    while True:
        dt = date(year, month, day)
        if dt.strftime("%a") == "Thu":
            break
        day += 1
    
    return day + 14 # add 2 weeks


def get_records(client, symbol, start_date, end_date):
    r"""
    Args:
        start_date, end_date: ddmmyyyy strings, inclusive.
    
    Returns:
        List of records.
    """

    records = []
    page = 1
    while page == 1 or len(ohlc_json["data"]) > 0:
        time.sleep(1)
        ohlc_json = client.intraday_ohlc(fc_data_config, model.intraday_ohlc(symbol, start_date, end_date, page, 1000))

        if (ohlc_json["message"] != "Success") or ("data" not in ohlc_json) or (ohlc_json["data"] == None):
            print(ohlc_json["message"])
        else:
            records += ohlc_json["data"]
            page += 1
    
    return records


def get_data():
    r"""
    Returns:
        2 lists of vn30f_records and vn30_records, range from 2020 to 2024 inclusively.
    """
    client = fc_md_client.MarketDataClient(fc_data_config)

    year = 2019
    month = 12
    end_day = get_third_thursday(year, month)

    vn30f_records = []
    vn30_records = []

    # loop for each month
    while year < 2025:
        start_day = end_day + 1
        start_date = date(year, month, start_day).strftime("%d/%m/%Y")

        month = month % 12 + 1
        if month == 1:
            year += 1

        end_day = get_third_thursday(year, month)
        end_date_dt = date(year, month, end_day)
        end_date = end_date_dt.strftime("%d/%m/%Y")

        symbol_vn30f = "vn30f" + end_date_dt.strftime("%y%m")

        print(f"{month}/{year}")

        # query
        # split query into 2 as the api only allow max range of 30 days
        tmp_date1 = date(year, month, 1).strftime("%d/%m/%Y")
        tmp_date2 = date(year, month, 2).strftime("%d/%m/%Y")
 
        vn30f_records += get_records(client, symbol_vn30f, start_date, tmp_date1)
        vn30f_records += get_records(client, symbol_vn30f, tmp_date2, end_date)

        vn30_records += get_records(client, "vn30", start_date, tmp_date1)
        vn30_records += get_records(client, "vn30", tmp_date2, end_date)
    
    print(f"VN30F: {len(vn30f_records)} records")
    print(f"VN30: {len(vn30_records)} records")

    return vn30f_records, vn30_records



def preprocess_data(vn30f_records, vn30_records):
    # Remove duplicate
    a = vn30f_records
    b = vn30_records
    vn30f_records = []
    vn30_records = []
    for i, x in enumerate(a):
        if i == 0 or x != a[i - 1]:
            vn30f_records.append(x)
    for i, x in enumerate(b):
        if i == 0 or x != b[i - 1]:
            vn30_records.append(x)


    record_of_day = 0
    current_date_str = ""
    next_expiration_date = date(1970, 1, 1)
    vn30f_ref_price = 0

    # Align records by time
    data = []
    i = 0
    j = 0
    x_ok = i < len(vn30f_records)
    y_ok = j < len(vn30_records)
    while x_ok or y_ok:
        print(i, j, end='\r')
        x = None
        y = None
        new_date = None

        if x_ok:
            x = vn30f_records[i]
        else:
            new_date = datetime.strptime(y["TradingDate"], "%d/%m/%Y").date()

        if y_ok:
            y = vn30_records[j]
        else:
            new_date = datetime.strptime(x["TradingDate"], "%d/%m/%Y").date()

        if x_ok and y_ok:
            new_date = min(datetime.strptime(x["TradingDate"], "%d/%m/%Y").date(),\
                            datetime.strptime(y["TradingDate"], "%d/%m/%Y").date())
        new_date_str = new_date.strftime("%d/%m/%Y")

        # Time in day
        if new_date_str == current_date_str:
            record_of_day += 1
        else:
            current_date_str = new_date_str
            record_of_day = 0

            # Update reference price
            vn30f_ref_price = x["Open"] if i == 0 else vn30f_records[i - 1]["Close"]


        # Days until next expiration
        current_date = new_date
        if current_date > next_expiration_date: # recalculate next expiration date
            next_month = current_date + timedelta(days=20)
            year = int(next_month.strftime("%Y"))
            month = int(next_month.strftime("%m"))
            day = get_third_thursday(year, month)
            next_expiration_date = date(year, month, day)

        days_until_expiration = (next_expiration_date - current_date).days
        day_of_year = (int(current_date.strftime("%j")) - 1)

        time = {
            "Expiration": days_until_expiration,
            "TimeOfDay": record_of_day,
            "DayOfYear": day_of_year
        }

        vn30f = None
        if x_ok and (
            not y_ok or\
            (x["TradingDate"] == current_date_str and x["Time"][:5] <= y["Time"][:5]) or\
            x["TradingDate"] != y["TradingDate"]
        ):
            vn30f = {
                "RefPrice": float(vn30f_ref_price),
                "Open": float(x["Open"]),
                "High": float(x["High"]),
                "Low": float(x["Low"]),
                "Close": float(x["Close"]),
                "Volume": int(x["Volume"]),
                "Mask": 1
            }
            i += 1
            x_ok = i < len(vn30f_records)
        else:
            vn30f = {
                "RefPrice": float(vn30f_ref_price), 
                "Mask": 0
            }

        vn30 = None
        if y_ok and (
            not x_ok or\
            (y["TradingDate"] == current_date_str and y["Time"][:5] <= x["Time"][:5]) or\
            y["TradingDate"] != x["TradingDate"]
        ):
            vn30 = {
                "Open": float(y["Open"]),
                "High": float(y["High"]),
                "Low": float(y["Low"]),
                "Close": float(y["Close"]),
                "Volume": int(y["Volume"]),
                "Mask": 1
            }
            j += 1
            y_ok = j < len(vn30_records)
        else:
            vn30 = {"Mask": 0}

        data.append({
            "Time": time,
            "VN30F": vn30f,
            "VN30": vn30,
        })

    return data

if __name__ == "__main__":
    vn30f_records, vn30_records = get_data()
    data = preprocess_data(vn30f_records, vn30_records)

    dir = os.path.dirname(config.data_path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    with open(config.data_path, "w") as fout:
        json.dump(data, fout)