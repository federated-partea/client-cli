import numpy as np
from numpy import full, nan
import pandas as pd


def compute(cond_col: str, data: pd.DataFrame, min_time: float = None, max_time: float = None, step_size: float = None):
    local_results = {}
    if cond_col:
        for category in data[cond_col].unique():
            df_col = data.loc[data[cond_col] == category]
            d_n_matrix = compute_d_and_n_matrix(df_col, min_time, max_time, step_size)
            local_results[category] = d_n_matrix.to_dict()
    else:
        d_n_matrix = compute_d_and_n_matrix(data, min_time, max_time, step_size)
        local_results["complete"] = d_n_matrix.to_dict()

    return local_results


def preprocess(duration_col: str, event_col: str, category_col: str, file_path: str, separator: str):
    columns = [duration_col, event_col]

    if category_col:
        columns.append(category_col)
    if separator == "spss":
        data = pd.read_sas(file_path, encoding='iso-8859-1').sort_values(by=duration_col)
        data = data.loc[:, columns]
    else:
        sep = ","
        if separator == "comma":
            sep = ","
        elif separator == "tab":
            sep = "\t"
        elif separator == "space":
            sep = " "

        data = pd.read_csv(file_path, sep=sep, usecols=columns).sort_values(by=duration_col)
        data[[duration_col, event_col]] = data[[duration_col, event_col]].astype("float")

    data = data.rename({duration_col: "time", event_col: "event_observed"}, axis=1)
    data = data.dropna()

    return data


def compute_d_and_n(data, t):
    temp = data[data["time"] == t].groupby("event_observed").count()
    try:
        d = temp.loc[1.0, "time"]
    except KeyError:
        d = 0
    try:
        n = temp.loc[0.0, "time"] + d
    except KeyError:
        n = 0 + d
    return d, n


def compute_d_and_n_matrix(data, min_time: float, max_time: float, step_size: float):
    if min_time is not None and max_time is not None and step_size is not None:
        timeline = np.arange(min_time, max_time, step_size)
    else:
        timeline = data["time"].sort_values().unique()
    di = full(len(timeline) + 1, nan)
    ni = full(len(timeline) + 1, nan)
    ni[0] = data.shape[0]
    for i in range(len(timeline)):
        d, n = compute_d_and_n(data, timeline[i])
        di[i] = d
        ni[i + 1] = ni[i] - n
    m = pd.DataFrame(index=timeline)
    m["di"] = di[:-1]
    m["ni"] = ni[:-1]

    return m
