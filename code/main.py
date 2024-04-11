import argparse
import os
import random
import tensorflow as tf
import dataProcess
import tensorflow.compat.v1 as tf
from Trainer import Trainer
from utils import process
import numpy as np

tf.reset_default_graph()
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # TensorFlow allocates video memory on demand
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Specify the allocation ratio of graphics memory





def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--min_cells', type=float, default=5,
                        help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    parser.add_argument('--Dim_PCA', type=int, default=1000, help='The output dimention of PCA')
    parser.add_argument('--Dim_PCA_net', type=int, default=50, help='The output dimention of PCA')
    parser.add_argument('--data_path', type=str, default='data/', help='The path to dataset')
    parser.add_argument('--data_name', type=str, default='DLPFC151509', help='The name of dataset')
    parser.add_argument('--Data_name', type=str, default='DLPFC',
                        help='Using mask for multi-dataset.')
    parser.add_argument('--DLPFC', type=str, default='151509', help='Slice Name')
    parser.add_argument('--preprocessed_data_path', type=str, default='processedData/',
                        help='The folder to store the preprocessed data')
    parser.add_argument('--highly_variable', type=int, default=2000)
    parser.add_argument('--highly_variable_or', type=int, default=1)
    parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
    parser.add_argument('--clusterOr', type=bool, default=False, help='Whether to determine the number of spatial domains')
    parser.add_argument('--n_neighbors', type=int, default=10, help='The number of neighbors used to determine the number of spatial domains')
    parser.add_argument('--res', type=int, default=1, help='Determine the number of spatial domain parameters')

    parser.add_argument('--dataset', nargs='?', default='DLPFC151509', help='Preprocessing input dataset')
    parser.add_argument('--input', nargs='?', default='input/', help='Input dataset')
    parser.add_argument('--result_img', nargs='?', default='result_img/', help='Visualization folder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dlr', type=float, default=0.0001, help='D Learning rate.')
    parser.add_argument('--n-epochs', default=155, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims1', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--hidden-dims2', type=list, nargs='+', default=[512, 512], help='Number of dimensions2.')
    parser.add_argument('--hidden-dims3', type=list, nargs='+', default=[512, 512], help='Number of dimensions3.')
    parser.add_argument('--lambda_', default=0.5, type=float)
    parser.add_argument('--dropout', default=0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=50.0, type=float, help='gradient clipping')
    parser.add_argument('--n_sample', default=4788, type=int, help='Number of spots')
    parser.add_argument('--random_state', default=2042, type=int, help='random_state')
    parser.add_argument('--cluster', default=7, type=int, help='Number of spatial domains')

    return parser.parse_args()


def main(args):
    if args.clusterOr:
        with open('processedData/'+args.dataset+'/'+args.input+'cluster.txt', 'r') as file:
            lines = file.readlines()
        arr = np.array(lines)
        arr=int(float(arr[0]))
        args.cluster=arr
    dataProcess.dataProcess(args)
    G1,G2,G3,X,Y,coordinates = process.load_data(args.dataset)
    print('Graph的维度：' + str(G1.shape))
    print('Content的维度：' + str(X.shape))
    Label = Y
    print('Label的维度：' + str(Label.shape))
    feature_dim1 = X.shape[1]
    args.hidden_dims1 = [feature_dim1] + args.hidden_dims1

    feature_dim2 = X.shape[1]
    args.hidden_dims2 = [feature_dim2] + args.hidden_dims2

    feature_dim3 = X.shape[1]
    args.hidden_dims3 = [feature_dim3] + args.hidden_dims3

    print('隐层单元1的维度：' + str(args.hidden_dims1))
    print('隐层单元2的维度：' + str(args.hidden_dims2))

    G_tf1, S1, R1 = process.prepare_graph_data(G1)
    G_tf2, S2, R2 = process.prepare_graph_data(G2)
    G_tf3, S3, R3 = process.prepare_graph_data(G3)

    result=[]
    result_fainal=[]
    for i in range(1,10):
        trainer = Trainer(args)
        _ = trainer.assign(G_tf1, X, S1, R1, G_tf2, X, S2, R2, G_tf3, X, S3, R3)
        fin = True
        trainer(G_tf1, X, S1, R1, G_tf2, X, S2, R2, G_tf3, X, S3, R3, Label, fin)
        finalAri = int(trainer.final_ari * 10000) / 10000
        result_fainal.append(finalAri)
        # tf.reset_default_graph()
        tf.keras.backend.clear_session()

    print(result)
    print(result_fainal)


if __name__ == "__main__":
    args = parse_args()
    seed=args.random_state
    random.seed(seed)  # set random seed for python
    np.random.seed(seed)  # set random seed for numpy
    tf.random.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    main(args)
