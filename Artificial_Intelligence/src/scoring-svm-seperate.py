import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy import stats


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k, v in iteritems(G)}


def main():
    parser = ArgumentParser("scoring",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--emb", required=True, help='Embeddings file')
    parser.add_argument("--network", required=True,
                        help='A .mat file containing the adjacency matrix and node labels of the input network.')
    parser.add_argument("--adj-matrix-name", default='network',
                        help='Variable name of the adjacency matrix inside the .mat file.')
    parser.add_argument("--label-matrix-name", default='group',
                        help='Variable name of the labels matrix inside the .mat file.')
    parser.add_argument("--num-shuffles", default=2, type=int, help='Number of shuffles.')
    parser.add_argument("--all", default=False, action='store_true',
                        help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
                             'By default, only training percents of 10, 50 and 90 are used.')

    args = parser.parse_args()
    # 0. Files
    embeddings_file = args.emb
    matfile = args.network

    # 1. Load Embeddings
    model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)

    # 2. Load labels
    mat = loadmat(matfile)
    A = mat[args.adj_matrix_name]
    graph = sparse2graph(A)
    labels_matrix = mat[args.label_matrix_name]
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))

    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(args.num_shuffles):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results_micro = defaultdict(list)
    all_results_macro = defaultdict(list)

    if args.all:
        training_percents = numpy.asarray(range(1, 10)) * .1
    else:
        training_percents = [0.1, 0.5, 0.9]

    tp_list = []

    for train_percent in training_percents:

        for shuf in shuffles:

            X, y = shuf
            y = y.todense()

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size, :]

            X_test = X[training_size:, :]
            y_test_ = y[training_size:, :]

            clf = SVC(gamma='auto')
            confusion_matrix_list = numpy.zeros((0, 4))

            precision_list = []
            recall_list = []

            for i in range(0, y.shape[1]):
                labels = numpy.unique(y_train_[:, i], axis=0)

                if labels.shape[0] < 2:
                    continue

                clf.fit(X_train, numpy.ravel(y_train_[:, i], order='C'))
                predicted = clf.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test_[:, i], predicted, labels=[0, 1]).ravel()

                confusion_matrix_list = numpy.append(confusion_matrix_list, [[tn, fp, fn, tp]], axis=0)

                if tp == 0:
                    precision = 0
                    recall = 0
                else:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)

                precision_list.append(precision)
                recall_list.append(recall)

                # if tp > 0:
                #     print(train_percent)
                #     print(tn, fp, fn, tp)
                #     tp_list.append(tp)

            sum_conf_mat_list = confusion_matrix_list.sum(axis=0)

            tn_sum = sum_conf_mat_list[0]
            fp_sum = sum_conf_mat_list[1]
            fn_sum = sum_conf_mat_list[2]
            tp_sum = sum_conf_mat_list[3]

            if tp_sum + fp_sum == 0:
                micro_average_precision = 0
            else:
                micro_average_precision = tp_sum / (tp_sum + fp_sum)

            if tp_sum + fn_sum == 0:
                micro_average_recall = 0
            else:
                micro_average_recall = tp_sum / (tp_sum + fn_sum)

            # micro_f1 = stats.hmean([micro_average_precision, micro_average_recall])

            if micro_average_precision + micro_average_recall == 0:
                micro_f1 = 0
            else:
                micro_f1 = 2 * micro_average_precision * micro_average_recall / (
                        micro_average_precision + micro_average_recall)

            macro_average_precision = sum(precision_list) / len(precision_list)
            macro_average_recall = sum(recall_list) / len(recall_list)

            # macro_f1 = stats.hmean([macro_average_precision, macro_average_recall])

            if macro_average_precision + macro_average_recall == 0:
                macro_f1 = 0
            else:
                macro_f1 = 2 * macro_average_precision * macro_average_recall / (
                        macro_average_precision + macro_average_recall)

            all_results_micro[train_percent].append(micro_f1)
            all_results_macro[train_percent].append(macro_f1)

    # print(all_results_micro)
    print("Micro")
    for key, value in all_results_micro.items():
        print(key, sum(value) / len(value))

    # print(all_results_micro)
    print("Macro")
    for key, value in all_results_macro.items():
        print(key, sum(value) / len(value))


if __name__ == "__main__":
    sys.exit(main())
