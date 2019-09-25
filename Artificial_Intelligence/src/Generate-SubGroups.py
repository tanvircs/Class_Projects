"""Separate the labels into 3 sets of categories and save as .mat files."""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def main():
    mat = scipy.io.loadmat('PPI-Data.mat')
    group = mat['group']
    network = mat['network']

    group_full_matrix = group.todense()
    sum_matrix = group_full_matrix.sum(axis=0)
    print(sum_matrix.shape)

    list_unbal = []
    list_medbal = []
    list_bal = []

    group_num = 20

    for i in range(0, 21979):
        if len(list_unbal) < group_num:
            if 18159 * 0.1 >= sum_matrix.item(i) > 18159 * 0.05:
                list_unbal.append(i)

        if len(list_medbal) < group_num:
            if 18159 * 0.3 >= sum_matrix.item(i) >= 18159 * 0.25:
                list_medbal.append(i)

        if len(list_bal) < group_num:
            if 18159 * 0.65 >= sum_matrix.item(i) >= 18159 * 0.35:
                list_bal.append(i)

    group_matrix_unbal = group_full_matrix[:, list_unbal]
    group_matrix_medbal = group_full_matrix[:, list_medbal]
    group_matrix_bal = group_full_matrix[:, list_bal]

    print(group_matrix_unbal.shape)
    print(group_matrix_medbal.shape)
    print(group_matrix_bal.shape)

    print(group_matrix_unbal.sum(axis=0))
    print(group_matrix_medbal.sum(axis=0))
    print(group_matrix_bal.sum(axis=0))

    sparse_group_matrix_unbal = sparse.csr_matrix(group_matrix_unbal)
    sparse_group_matrix_medbal = sparse.csr_matrix(group_matrix_medbal)
    sparse_group_matrix_bal = sparse.csr_matrix(group_matrix_bal)

    scipy.io.savemat('output/PPI-network-unbalance.mat', mdict={'network': network, 'group': sparse_group_matrix_unbal})
    scipy.io.savemat('output/PPI-network-medium_balance.mat',
                     mdict={'network': network, 'group': sparse_group_matrix_medbal})
    scipy.io.savemat('output/PPI-network-balance.mat', mdict={'network': network, 'group': sparse_group_matrix_bal})


if __name__ == "__main__":
    main()
