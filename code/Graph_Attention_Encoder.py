import tensorflow.compat.v1 as tf
from sklearn.cluster import KMeans


class GATE():
    def __init__(self, hidden_dims1, hidden_dims2, hidden_dims3, lambda_,n_clusters,n_sample,random_state):
        self.lambda_ = lambda_
        self.n_layers1 = len(hidden_dims1) - 1
        self.n_layers2 = len(hidden_dims2) - 1
        self.n_layers3 = len(hidden_dims3) - 1
        self.W, self.v = self.define_weights(hidden_dims1)
        self.C = {}
        self.W2, self.v2 = self.define_weights2(hidden_dims2)
        self.C2 = {}
        self.W3, self.v3 = self.define_weights3(hidden_dims3)
        self.C3 = {}
        self.params = {"n_clusters": n_clusters, "encoder_dims": [512], "alpha": 1.0, "n_sample": n_sample}
        self.mu = tf.Variable(tf.zeros(shape=(self.params["n_clusters"], 128)), name="mu")#128
        self.kmeans = KMeans(n_clusters=self.params["n_clusters"], n_init=20,random_state=random_state)
        self.n_cluster = self.params["n_clusters"]
        self.input_batch_size = n_sample
        self.alpha = self.params['alpha']


    def __call__(self, A, X, R, S, p, A2, X2, R2, S2, A3, X3, R3, S3):
        # Encoder1
        H = X
        for layer in range(self.n_layers1):
            H = self.__encoder(A, H, layer)

        self.H = H

        # Decoder1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H

        self.z = self.__dense_layer(self.H)
        print('z的维度' + str(self.z.shape))
        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder2(A2, H2, layer)

        self.H2 = H2

        # Decoder2
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder2(H2, layer)
        X_2 = H2

        self.z2 = self.__dense_layer2(self.H2)

        # Encoder3
        H3 = X3
        for layer in range(self.n_layers3):
            H3 = self.__encoder3(A3, H3, layer)

        self.H3 = H3

        # Decoder3
        for layer in range(self.n_layers3 - 1, -1, -1):
            H3 = self.__decoder3(H3, layer)
        X_3 = H3

        self.z3 = self.__dense_layer3(self.H3)

        with tf.name_scope("distribution"):
            self.q = self._soft_assignment((0.9 * self.z + 0.04 * self.z2 + 0.06 * self.z3), self.mu)

            self.p = p
            self.pred = tf.argmax(self.q, axis=1)

        # The reconstruction loss of node features
        features_loss = tf.reduce_mean((X - X_) ** 2) + tf.reduce_mean((X2 - X_2) ** 2)+ tf.reduce_mean((X3 - X_3) ** 2)

        self.C_loss = self._kl_divergence(self.p, self.q)
        # The loss of Similarity measurement

        self.dense_loss = 0.0001 * tf.reduce_mean((2*self.z - self.z2 - self.z3) ** 2)

        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        self.S_emb2 = tf.nn.embedding_lookup(self.H2, S2)
        self.R_emb2 = tf.nn.embedding_lookup(self.H2, R2)
        self.S_emb3 = tf.nn.embedding_lookup(self.H3, S3)
        self.R_emb3 = tf.nn.embedding_lookup(self.H3, R3)

        # Total loss
        self.loss = features_loss + 5 * self.C_loss + self.dense_loss
        embeding = 0.9 * self.z + 0.04 * self.z2 + 0.06 * self.z3
        return self.loss, self.H, self.C, self.H2, self.C2, self.H3, self.C3, self.pred, self.dense_loss, self.z, self.z2, self.z3, features_loss, embeding


    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __encoder2(self, A, H, layer):
        H = tf.matmul(H, self.W2[layer])
        self.C2[layer] = self.graph_attention_layer2(A, H, self.v2[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __decoder2(self, H, layer):
        H = tf.matmul(H, self.W2[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __encoder3(self, A, H, layer):
        H = tf.matmul(H, self.W3[layer])
        self.C3[layer] = self.graph_attention_layer3(A, H, self.v3[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C3[layer], H)

    def __decoder3(self, H, layer):
        H = tf.matmul(H, self.W3[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C3[layer], H)


    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers1):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def define_weights2(self, hidden_dims):
        W2 = {}
        for i in range(self.n_layers2):
            W2[i] = tf.get_variable("W2%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att2 = {}
        for i in range(self.n_layers2):
            v2 = {}
            v2[0] = tf.get_variable("v2%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v2[1] = tf.get_variable("v2%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att2[i] = v2

        return W2, Ws_att2

    def define_weights3(self, hidden_dims):
        W3 = {}
        for i in range(self.n_layers3):
            W3[i] = tf.get_variable("W3%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att3 = {}
        for i in range(self.n_layers3):
            v3 = {}
            v3[0] = tf.get_variable("v3%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v3[1] = tf.get_variable("v3%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att3[i] = v3

        return W3, Ws_att3

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def graph_attention_layer2(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def graph_attention_layer3(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def get_assign_cluster_centers_op(self, features):

        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        print(embeddings.shape)
        print(cluster_centers.shape)
        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        print("1")
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    # KL
    def _kl_divergence(self, target, pred):
        print(target.shape)
        print(pred.shape)
        return tf.reduce_mean((target - pred) ** 2)


    def __dense_layer(self, Z):
        dense1 = tf.layers.dense(inputs=Z, units=512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense2, units=128, activation=None)#128
        return logits
    def __dense_layer2(self, Z2):
        dense1 = tf.layers.dense(inputs=Z2, units=512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
        logits2 = tf.layers.dense(inputs=dense2, units=128, activation=None)
        return logits2
    def __dense_layer3(self, Z3):
        dense1 = tf.layers.dense(inputs=Z3, units=512, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
        logits3 = tf.layers.dense(inputs=dense2, units=128, activation=None)
        return logits3
