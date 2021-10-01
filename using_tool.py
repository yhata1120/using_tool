import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import Kfold

def pre_Kfolded_dfdata(train_indices, val_indices):
    X_train_indices, X_val_indices = name_list[train_indices], name_list[val_indices]
    X_train_indices = list([tuple(e) for e in X_train_indices])
    X_val_indices = list([tuple(e) for e in X_val_indices])
    group_train = pd.concat([data_grouped.get_group(group_name) for group_name in X_train_indices])
    group_val = pd.concat([data_grouped.get_group(group_name) for group_name in X_val_indices])
    # returnされるのは訓練用データが入ったデータフレームと検証用データが入ったデータフレーム
    return group_train, group_val


def split_df(data_tot, group_level = ["Ag","axis","gbindex",]):  # by [element, axis, gbindex]
    data_grouped = data_tot.groupby(by=group_level) 
    name_list = np.array(list(data_grouped.indices.keys()))
    return data_grouped, name_list


# data_grouped, name_list = split_df(data_tot, group_level = ["Ag","axis","gbindex"])

# for fold, (train_indices, val_indices) in enumerate(kf.split(name_list)):
#     group_train, group_val = pre_Kfolded_dfdata(train_indices,val_indices)
#     print(group_val)