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

from numpy import cross
from interface_generator import unit_v
def get_conversion_matrix(interface, rotation_axis = np.array([0,0,0,])):

    # interface.orient(this orientation表示(y方向はlattice_1のy方向と反対)→simulation_cell表示) 
    # → this_orientation(simulation_cell表示→this orientation表示(y方向はlattice_1のy方向と反対))
    # this orientation = three basis of cell of representation of lattice_1
    # cellでのx,y,zはこの3つの基底の方向に沿った移動量を表している
    this_orientation = inv(interface.orient)
    # get three vectors that specify cellx, celly, and cellz,direcition in lattice_1 coordination
    v1, v2, v3 = this_orientation.T

    gb_indice = v1 # should be correspond to v1 of this orientation
    rotation_axis = unit_v(rotation_axis) # should be correspond to direction of rotation axis
    third_direction = cross(gb_indice, rotation_axis) # cross of gb_indice and rotation_axis
    
    cell_basis = this_orientation
    # セルのxyzの基底をlattice_1の右手系conventional cell表示で表した

    standard_scaled_basis = np.column_stack((gb_indice,rotation_axis,third_direction))
    # 自分が基準にしている3方向をlattice_1の右手系conventional cell表示で表した
    
    # cellのx,y,z方向を決める基底に沿った量を自分の基準とする基底の量に換算(lattice_1conv表示)
    conversion_matrix = np.dot(inv(standard_scaled_basis),cell_basis)
    conversion_matrix = np.array([[conversion_matrix[1,1,],conversion_matrix[1,2,]],
                                 [conversion_matrix[2,1,],conversion_matrix[2,2,],]])
    return conversion_matrix

def extract_descriptor(df):
    X_all = df[['misorientation','dy','dz','sigma','gbarea','1stnnmean','2ndnnmean','shorter_bonds_num','longer_bonds_num','shorter_bonds_mean','longer_bonds_mean','mininum_bond','num_dangling_bond','num_aroundgb']]
    X_all.loc[:,'dangling_bond/gbarea'] = X_all.loc[:,'num_dangling_bond']/X_all.loc[:,'gbarea']
    X_all.loc[:,'shorter_bonds/gbarea'] = X_all.loc[:,'shorter_bonds_num']/X_all.loc[:,'gbarea']
    X_all.loc[:,'longer_bonds/gbarea'] = X_all.loc[:,'longer_bonds_num']/X_all.loc[:,'gbarea']
    X_all.loc[:,'num_aroundgb/gbarea'] = X_all.loc[:,'num_aroundgb']/X_all.loc[:,'gbarea']
    X_all.loc[:,'tan'] = np.tan(X_all.loc[:,'misorientation']/180*np.pi/2)
    X_all.drop(['misorientation', 'shorter_bonds_num','longer_bonds_num','num_dangling_bond','gbarea','num_aroundgb'], axis=1,inplace = True)
    X_nonzero = X_all.copy()
    X_nonzero.drop(['dy','dz','shorter_bonds/gbarea','longer_bonds/gbarea','dangling_bond/gbarea','tan'], axis=1,inplace = True)
    X_nonzero_array = X_nonzero.values
    X_all_array = X_all.values
    X_all_square = X_all_array**2
    X_inv = 1/X_nonzero_array
    X_all_sqinv = X_inv**2
    X_all_exp = np.exp(X_all_array)
    X_all_invexp = np.exp(X_inv)
    X_all_tot = np.concatenate([X_all_array,X_all_square,X_inv,X_all_sqinv,X_all_exp,X_all_invexp,],1)
    y_all = df[['energy','delta_x']].values
    return X_all_tot, y_all


def calc_frantional(cnids):
    cnids.loc[:,"invcnidy1"] = cnids.loc[:,"cnidv2z"]/(cnids.loc[:,"cnidv1y"]*cnids.loc[:,"cnidv2z"]-cnids.loc[:,"cnidv1z"]*cnids.loc[:,"cnidv2y"])
    cnids.loc[:,"invcnidz1"] = -cnids.loc[:,"cnidv2y"]/(cnids.loc[:,"cnidv1y"]*cnids.loc[:,"cnidv2z"]-cnids.loc[:,"cnidv1z"]*cnids.loc[:,"cnidv2y"])
    cnids.loc[:,"invcnidy2"] = -cnids.loc[:,"cnidv1z"]/(cnids.loc[:,"cnidv1y"]*cnids.loc[:,"cnidv2z"]-cnids.loc[:,"cnidv1z"]*cnids.loc[:,"cnidv2y"])
    cnids.loc[:,"invcnidz2"] = cnids.loc[:,"cnidv1y"]/(cnids.loc[:,"cnidv1y"]*cnids.loc[:,"cnidv2z"]-cnids.loc[:,"cnidv1z"]*cnids.loc[:,"cnidv2y"])
    cnids.loc[:,"cnid1"] = cnids.loc[:,"invcnidy1"]*cnids.loc[:,"dy"] + cnids.loc[:,"invcnidz1"]*cnids.loc[:,"dz"]
    cnids.loc[:,"cnid2"] = cnids.loc[:,"invcnidy2"]*cnids.loc[:,"dy"] + cnids.loc[:,"invcnidz2"]*cnids.loc[:,"dz"]
    cnids.loc[:,"sincnid1"] = np.sin(np.pi*cnids.loc[:,"cnid1"])
    cnids.loc[:,"sincnid2"] = np.sin(np.pi*cnids.loc[:,"cnid2"])

