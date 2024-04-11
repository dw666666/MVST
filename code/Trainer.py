import tensorflow.compat.v1 as tf
from Graph_Attention_Encoder import GATE
from utils import process
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score,fowlkes_mallows_score
import numpy as np


nmi = normalized_mutual_info_score
ari = adjusted_rand_score
fmi = fowlkes_mallows_score

def b3_precision_recall_fscore(labels_true, labels_pred):
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")


    n_samples = len(labels_true)
    true_clusters = {}
    pred_clusters = {}

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def f_score(labels_true, labels_pred):
    _, _, f = b3_precision_recall_fscore(labels_true, labels_pred)
    return f

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    sum1 = 0
    for i in range(ind[0].size-1):
        sum1 += w[ind[0][i],ind[1][i]]

    return sum1 * 1.0 / y_pred.size


class Trainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders(args)
        self.gate = GATE(args.hidden_dims1, args.hidden_dims2,args.hidden_dims3, args.lambda_, args.cluster, args.n_sample, args.random_state)
        self.loss, self.H, self.C, self.H2, self.C2, self.H3, self.C3, self.pred, self.dense_loss, self.z, self.z2, self.z3, self.features_loss,self.embeding = self.gate(
            self.A, self.X, self.R, self.S, self.p, self.A2, self.X2, self.R2, self.S2, self.A3, self.X3, self.R3,self.S3)
        self.optimize(self.loss)
        self.build_session()
        self.best_ari = 0
        self.best_embeding = np.zeros((1, 128))
        self.best_pred = np.zeros((1, args.n_sample))
        self.final_ari = 0
        self.final_embeding = np.zeros((1, 128))
        self.final_pred = np.zeros((1, args.n_sample))

    def build_placeholders(self, args):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)
        self.A2 = tf.sparse_placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.S2 = tf.placeholder(tf.int64)
        self.R2 = tf.placeholder(tf.int64)
        self.A3 = tf.sparse_placeholder(dtype=tf.float32)
        self.X3 = tf.placeholder(dtype=tf.float32)
        self.S3 = tf.placeholder(tf.int64)
        self.R3 = tf.placeholder(tf.int64)
        self.p = tf.placeholder(tf.float32, shape=(None, args.cluster))


    def build_session(self, gpu=True):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not gpu:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3, L, fin=False):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3, L, fin)

    def run_epoch(self, epoch, A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3, L, fin):
        q = self.session.run(self.gate.q, feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2, self.A3: A3, self.X3: X3, self.S3: S3, self.R3: R3})
        p = self.gate.target_distribution(q)

        if not fin:
            loss, pred, _, st_loss, f_loss, d_loss, embeding = self.session.run([self.loss, self.pred, self.train_op, self.structure_loss, self.features_loss, self.dense_loss, self.embeding],
                                             feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2, self.A3: A3, self.X3: X3, self.S3: S3, self.R3: R3})
            if self.best_ari < ari(L, pred):
                self.best_ari = ari(L, pred)
                self.best_embeding = embeding
                self.best_pred = pred
            self.final_pred=pred
            self.final_embeding=embeding
            self.final_ari=ari(L, pred)

            if epoch % 5 == 0:
                print(
                    "Epoch--{}:\tloss: {:.8f}\t\tsloss: {:.8f}\t\tfloss: {:.8f}\t\tdloss: {:.8f}\t\nacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}\t\tfmi: {:.8f}".
                    format(epoch, loss, st_loss, f_loss, d_loss, cluster_acc(L, pred), nmi(L, pred), f_score(L, pred),
                           ari(L, pred), fmi(L, pred)))
                print('best ari : '+str(self.best_ari))
                print('final ari : ' + str(self.final_ari))


        elif fin:
            loss, pred, _, f_loss, d_loss, embeding = self.session.run(
                [self.loss, self.pred, self.train_op, self.features_loss, self.dense_loss, self.embeding],
                feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p, self.A2: A2, self.X2: X2, self.S2: S2,
                           self.R2: R2, self.A3: A3, self.X3: X3, self.S3: S3, self.R3: R3})
            if self.best_ari < ari(L, pred):
                self.best_ari = ari(L, pred)
                self.best_embeding = embeding
                self.best_pred = pred
            self.final_pred = pred
            self.final_embeding = embeding
            self.final_ari = ari(L, pred)

            if epoch % 5 == 0:
                print(
                    "Epoch--{}:\tloss: {:.8f}\t\tfloss: {:.8f}\t\tdloss: {:.8f}\t\nacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}\t\tfmi: {:.8f}".
                    format(epoch, loss, f_loss, d_loss, cluster_acc(L, pred), nmi(L, pred), f_score(L, pred),
                           ari(L, pred), fmi(L, pred)))
                print('best ari : '+str(self.best_ari))
                print('final ari : ' + str(self.final_ari))


    def infer(self, A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3):
        H, C, z, H2, C2, z2, H3, C3, z3 = self.session.run([self.H, self.C, self.z, self.H2, self.C2, self.z2, self.H3, self.C3, self.z3],
                                feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2, self.A3: A3, self.X3: X3, self.S3: S3, self.R3: R3})
        return H, process.conver_sparse_tf2np(C), z, H2, process.conver_sparse_tf2np(C2), z2, H3, process.conver_sparse_tf2np(C3), z3

    def assign(self, A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3):
        _, _, embeddings, _, _, embeddings2, _, _, embeddings3 = self.infer(A, X, S, R, A2, X2, S2, R2, A3, X3, S3, R3)
        assign_mu_op = self.gate.get_assign_cluster_centers_op((0.9 * embeddings + 0.04 * embeddings2 + 0.06 * embeddings3))
        _ = self.session.run(assign_mu_op)