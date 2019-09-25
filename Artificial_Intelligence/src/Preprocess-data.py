"""Preprocess data using the raw data set and build the label matrix and network matrix and
save in one mat file as two variables. Also it is removing unnecessary nodes those do not have any labels associated."""

import numpy as np
import scipy.io
from scipy import sparse
import os


def main():
    group_file = "human.GO_2_string.2018.tsv"
    links_file = "9606.protein.links.v11.0.txt"

    set1 = set()
    set2 = set()
    with open(group_file) as f:
        next(f)
        for line in f:
            split_line = line.split()
            set1.add(split_line[3])

    with open(links_file) as f:
        next(f)
        for line in f:
            split_line = line.split()

            node1 = split_line[0]
            node2 = split_line[1]

            if node1 in set1 and node2 in set1:
                set2.add(node1)
                set2.add(node2)

    group = {}
    label_set = set()
    with open(group_file) as f:
        next(f)
        for line in f:
            split_line = line.split()
            node = split_line[3]

            if node not in set2:
                continue

            label = split_line[2]
            label_set.add(label)

            if node in group:
                labels = group[node]
                labels.append(label)
            else:
                labels = [label]
                group[node] = labels

    label_list = list(label_set)
    group_matrix = np.zeros((len(group.keys()), len(label_list)))
    group_keys = list(group.keys())

    # create directory if not exists
    if not os.path.exists("output"):
        os.makedirs("output")

    output_node_file = os.path.join("output/node-list.txt")
    output_label_file = os.path.join("output/label-list.txt")
    out_node = open(output_node_file, "w")
    for n in group_keys:
        out_node.write("{}\n".format(n))
    out_node.close()

    out_label = open(output_label_file, "w")
    for l in label_list:
        out_label.write("{}\n".format(l))
    out_label.close()

    for key, value in group.items():
        for v in value:
            group_matrix[group_keys.index(key)][label_list.index(v)] = 1

    sparse_group = sparse.csr_matrix(group_matrix)

    network = np.zeros((len(group.keys()), len(group.keys())))

    with open(links_file) as f:
        next(f)
        for line in f:
            split_line = line.split()

            node1 = split_line[0]
            node2 = split_line[1]

            if node1 in group_keys and node2 in group_keys:
                network[group_keys.index(node1)][group_keys.index(node2)] = 1

    sparse_network = sparse.csr_matrix(network)
    scipy.io.savemat('output/PPI-Data.mat', mdict={'network': sparse_network, 'group': sparse_group})


if __name__ == '__main__':
    main()
