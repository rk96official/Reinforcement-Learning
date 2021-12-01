import tensorflow as tf
import numpy as np


class score_model(object):

    def __init__(self, params, alpha=None, beta=None):
        self.epochs = params['epochs']
        self.learning_rate = params['lr']
        self.l2_reg = params['l2_reg']
        self.path_to_model = "./tmp/model.ckpt"
        self.print_freq = params['print_freq']
        self.alpha = alpha
        self.beta = beta
        return

    def fit(self, train_x, train_y, emb_u=None, emb_r=None, test_x=None, test_y=None, path_to_model=None):

        if self.alpha is None or self.beta is None:
            alpha, beta = self._compute_init_values(emb_u, emb_r)
            self.alpha = alpha
            self.beta = beta

        tf.reset_default_graph()
        m = np.shape(train_x)[1]
        emb_dim = np.shape(train_x)[2]
        self.m = m
        self.emb_dim = emb_dim
        train_y = train_y.reshape(m)

        if test_y is not None:
            test_y = test_y.reshape(np.shape(test_x)[1])

        M = tf.Variable(tf.eye(emb_dim), dtype=tf.float32)
        x = tf.placeholder(tf.float32, shape=[2, None, emb_dim], name='x')
        y = tf.placeholder(tf.float32, shape=[None], name='y')
        emb_context = tf.cast(x[0, :, :], tf.float32, name='emb_c')  # m * emb_dim
        emb_response = tf.cast(x[1, :, :], tf.float32, name='emb_r')
        prod_in = tf.tensordot(emb_response, M, axes=1)  # prod_in shape: (m,50)
        prod_out = tf.multiply(emb_context, prod_in)
        pred = tf.reduce_sum(prod_out, axis=1, name="pred_to_restore")
        output = (pred - self.alpha) / self.beta
        regularizer = tf.nn.l2_loss(M)
        loss = tf.reduce_mean(tf.square(output - y)) + self.l2_reg * regularizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            for i in list(range(self.epochs)):
                session.run(optimizer, feed_dict={x: train_x, y: train_y})
                if i % self.print_freq == 0:
                    if test_x is not None:
                        pred_train = session.run(output, feed_dict={x: train_x, y: train_y})
                        pred_test = session.run(output, feed_dict={x: test_x, y: test_y})
                        train_acc = self.get_accuracy(pred_train, train_y)
                        test_acc = self.get_accuracy(pred_test, test_y)
                        print('Iteration %d loss %f, train acc: %f, test acc %f, pred: %f, %f, %f, %f' % (
                            i, session.run(loss, feed_dict={x: train_x, y: train_y}), train_acc, test_acc,
                            session.run(tf.reduce_min(output), feed_dict={x: train_x}),
                            session.run(tf.reduce_max(output), feed_dict={x: train_x}),
                            session.run(tf.reduce_min(pred), feed_dict={x: train_x}),
                            session.run(tf.reduce_max(pred), feed_dict={x: train_x})))
                    else:
                        pred_train = session.run(output, feed_dict={x: train_x, y: train_y})
                        train_acc = self._get_accuracy(pred_train, train_y)
                        print('Iteration %d loss %f, train acc: %f' % (
                            i, session.run(loss, feed_dict={x: train_x, y: train_y}), train_acc))
            pred_train = session.run(output, feed_dict={x: train_x})

            if test_x is not None:
                pred_test = session.run(output, feed_dict={x: test_x})
                test_acc = self.get_accuracy(pred_test, test_y)
            else:
                test_acc = None
            train_acc = self._get_accuracy(pred_train, train_y)
        return train_acc, test_acc

    def _compute_init_values(self, emb_u, emb_r):
        n_samples = np.shape(emb_u)[0]
        prod_list = []

        for i in range(n_samples):
            term = 0
            term += np.dot(emb_u[i, :], emb_r[i, :])
            prod_list.append(term)

        alpha = np.min(prod_list)
        beta = max(prod_list) - min(prod_list)
        return alpha, beta

    def _get_accuracy(self, prediction, target, target_cont=True):
        pred = np.where(prediction >= 0.5, 1, 0)
        if target_cont:
            target = np.where(target >= 0.5, 1, 0)
            errors = np.subtract(pred, target)
            accuracy = list(errors).count(0) / float(np.size(errors))
            return accuracy

    def predict(self, test_x, path_to_model=None):
        if path_to_model is None:
            path_to_model = self.path_to_model

        path_meta = path_to_model + '.meta'
        path_ckp = path_to_model.replace(path_to_model.split('/')[-1], '')

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            saver = tf.train.import_meta_graph(path_meta)
            saver.restore(session, tf.train.latest_checkpoint(path_ckp))
            graph_r = tf.get_default_graph()
            x = graph_r.get_tensor_by_name("x:0")
            output = graph_r.get_tensor_by_name("pred_to_restore:0")
            prediction = session.run(output, feed_dict={x: test_x})
            pred_test = (prediction - self.alpha) / self.beta
        return pred_test

