"""Draw the histogram for the labels."""

import scipy.io
import matplotlib.pyplot as plt


def main():
    mat = scipy.io.loadmat('PPI-Data.mat')
    group = mat['group']

    label_size = group.get_shape()[1]
    cx = group.tocoo()
    list_group_sum = [0] * label_size
    for i, j, v in zip(cx.row, cx.col, cx.data):
        list_group_sum[j] += 1

    plt.title("Histogram")
    plt.hist(list_group_sum, bins=1000)
    plt.show()


if __name__ == "__main__":
    main()
