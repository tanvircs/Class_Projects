import scipy.io
import os


# this is used to generate the input files for LINE and Node2vec
def generate_input_files():
    mat = scipy.io.loadmat('PPI-Data.mat')
    network = mat['network']

    output_dir = "output"

    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "LINE-input.txt")
    out = open(output_file, "w")

    output_file2 = os.path.join(output_dir, "Node2vec-input.txt")
    out2 = open(output_file2, "w")

    cx = network.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        out.write("{} {} {}\n".format(i, j, int(v)))
        out2.write("{} {}\n".format(i, j))

    out.close()
    out2.close()


# call the function needed for the use case
if __name__ == '__main__':
    generate_input_files()
