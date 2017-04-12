"""This script is the FishFinder graph for the Kaggle Nature Conservancy Fishery
Competition.  It utilizes VGG-19 model as pre-trained weights during the
convolution steps"""


fishyfish = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(np.load(pretrained_path+'W_conv_1.npy'), trainable = False)
                b_conv1 = tf.Variable(np.load(pretrained_path+'b_conv_1.npy'), trainable = False)
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(np.load(pretrained_path+'W_conv_2.npy'), trainable = False)
                b_conv2 = tf.Variable(np.load(pretrained_path+'b_conv_2.npy'), trainable = False)
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(np.load(pretrained_path+'W_conv_3.npy'), trainable = False)
                b_conv3 = tf.Variable(np.load(pretrained_path+'b_conv_3.npy'), trainable = False)
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(np.load(pretrained_path+'W_conv_4.npy'), trainable = False)
                b_conv4 = tf.Variable(np.load(pretrained_path+'b_conv_4.npy'), trainable = False)
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(np.load(pretrained_path+'W_conv_5.npy'), trainable = False)
                b_conv5 = tf.Variable(np.load(pretrained_path+'b_conv_5.npy'), trainable = False)
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(np.load(pretrained_path+'W_conv_6.npy'), trainable = False)
                b_conv6 = tf.Variable(np.load(pretrained_path+'b_conv_6.npy'), trainable = False)
                tf.summary.histogram('W_conv6', W_conv6)
                tf.summary.histogram('b_conv6', b_conv6)
        with tf.variable_scope('Dense_layers') :
            with tf.name_scope('dense_1') :
                W_fc1 = tf.Variable(tf.truncated_normal([nodes_after_conv, fc_depth[0]], stddev = stddev ))
                b_fc1 = tf.Variable(tf.zeros([fc_depth[0]]))
                tf.summary.histogram('W_fc1', W_fc1)
                tf.summary.histogram('b_fc1', b_fc1)
            with tf.name_scope('dense_2') :
                W_fc2 = tf.Variable(tf.truncated_normal([fc_depth[0], fc_depth[1]], stddev = stddev ))
                b_fc2 = tf.Variable(tf.zeros([fc_depth[1]]))
                tf.summary.histogram('W_fc2', W_fc2)
                tf.summary.histogram('b_fc2', b_fc2)
            with tf.name_scope('dense_3') :
                W_fc3 = tf.Variable(tf.truncated_normal([fc_depth[1], fc_depth[2]], stddev = stddev ))
                b_fc3 = tf.Variable(tf.zeros([fc_depth[2]]))
                tf.summary.histogram('W_fc3', W_fc3)
                tf.summary.histogram('b_fc3', b_fc3)
            with tf.name_scope('dense_4') :
                W_fc4 = tf.Variable(tf.truncated_normal([(fc_depth[2]+32+1), fc_depth[3]], stddev = stddev ))
                b_fc4 = tf.Variable(tf.zeros([fc_depth[3]]))
                tf.summary.histogram('W_fc4', W_fc4)
                tf.summary.histogram('b_fc4', b_fc4)
        with tf.variable_scope('Classifiers') :
            with tf.name_scope('FishNoF') :
                W_clf = tf.Variable(tf.truncated_normal([fc_depth[3],num_labels], stddev = stddev))
                b_clf = tf.Variable(tf.zeros([num_labels]))
                tf.summary.histogram('W_clf', W_clf)
                tf.summary.histogram('b_clf', b_clf)


    def convolutions(data) :
        """
        Emulates VGG-19 architecture.
        """
        with tf.name_scope('Convolution') :
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(data, filter = W_conv1,
                                    strides = [1, 2, 2, 1],
                                    padding = 'SAME') + b_conv1)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv2,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv2),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv3,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv3)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv4,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv4),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv5,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv5)

            conv_layer = tf.nn.max_pool(tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv6,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv6),
                                ksize = [1, pool_kernel, pool_kernel,1],
                                strides = [1, pool_stride, pool_stride, 1],
                                padding ='VALID')
        return conv_layer

    def dense_layers(data, embedding, FiNoF, keep_prob) :
        """
        Executes a series of dense layers.
        """
        def fc(data, W, b, keep_prob = keep_prob) :
            """Convenience function for dense layer with dropout"""
            fc = tf.nn.dropout(
                    tf.nn.tanh(
                            tf.matmul(data, W) + b,
                            ),
                    keep_prob)
            return fc

        fc_layer = fc(data, W_fc1, b_fc1, keep_prob[0])
        fc_layer = fc(fc_layer, W_fc2, b_fc2,  keep_prob[1])
        fc_layer = fc(fc_layer, W_fc3, b_fc3, keep_prob[2])
        fc_layer = tf.concat(1, [fc_layer, embedding, FiNoF])
        fc_layer = fc(fc_layer, W_fc4, b_fc4, keep_prob[3])
        return fc_layer




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :

            fovea = tf.placeholder(tf.float32, shape = [batch_size, 64, 64, num_channels])
            embedding = tf.placeholder(tf.float32, shape = [batch_size, 32])
            fish_prob = tf.placeholder(tf.float32, shape = [batch_size, 1])
            labels = tf.placeholder(tf.float32, shape = [batch_size,num_labels])
            fovea_weights = tf.placeholder(tf.float32, shape = [batch_size, 1])
            label_weights = tf.placeholder(tf.float32, shape = [batch_size, 1])


            gamma_fovea = tf.placeholder(tf.float32, shape = ())
            gamma_label = tf.placeholder(tf.float32, shape = ())
            learning_rate = tf.placeholder(tf.float32, shape = () )
            beta_regularizer = tf.placeholder(tf.float32, shape = ())
        with tf.name_scope('Network') :
            conv_output = convolutions(fovea)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, embedding, fish_prob, keep_prob = keep_prob)
        with tf.name_scope('Classifiers') :
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation') :
            xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits = logits, labels = labels)
            fovea_cost = gamma_fovea*tf.reduce_mean(tf.multiply(xent, fovea_weights))
            label_cost = gamma_label*tf.reduce_mean(tf.multiply(xent, label_weights))

            regularization_term = (tf.nn.l2_loss(W_fc4) +
                                   tf.nn.l2_loss(W_fc3) +
                                   tf.nn.l2_loss(W_fc2) +
                                   tf.nn.l2_loss(W_fc1)
                                  ) * beta_regularizer

            cost = tf.reduce_mean(xent) + fovea_cost + label_cost + regularization_term

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)


    with tf.name_scope('Validation') :
        with tf.name_scope('Input') :

            val_fovea = tf.placeholder(tf.float32, shape = [valid_size, 64, 64, num_channels])
            val_embedding = tf.placeholder(tf.float32, shape = [valid_size, 32])
            val_fish_prob = tf.placeholder(tf.float32, shape = [valid_size, 1])
            val_labels = tf.placeholder(tf.float32, shape = [valid_size, num_labels])

        with tf.name_scope('Network') :
            v_conv_output = convolutions(val_fovea)
            v_dense_input = tf.contrib.layers.flatten(v_conv_output)
            v_dense_output = dense_layers(v_dense_input, val_embedding, val_fish_prob, keep_prob = [1.0, 1.0, 1.0, 1.0])
        with tf.name_scope('Classifiers') :
            val_logits = tf.matmul(v_dense_output, W_clf) + b_clf
        with tf.name_scope('Metrics') :
            val_loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits = val_logits, labels = val_labels)
            val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_logits, 1), tf.argmax(val_labels, 1)), dtype = tf.float32))

    with tf.name_scope('Summaries') :
        with tf.name_scope('Valid_Set') :
            tf.summary.scalar('Accuracy_FiNoF', val_acc)
            tf.summary.scalar('CrossEntropy', val_loss)
        with tf.name_scope('Parameters') :
            tf.summary.scalar('LearningRate', learning_rate)
            tf.summary.scalar('Regularization_Coefficient', beta_regularizer)
            tf.summary.scalar('Fovea_Cost_Coefficient', gamma_fovea)
            tf.summary.scalar('Label_Cost_Coefficient', gamma_label)

        with tf.name_scope('Train_Set') :
            tf.summary.scalar("Cost", cost)
            tf.summary.scalar('Regularization', regularization_term)
            tf.summary.scalar('Fovea_cost', fovea_cost)
            tf.summary.scalar('Label_cost', label_cost)
        summaries = tf.summary.merge_all()


    with tf.name_scope('Prediction') :
        with tf.name_scope('Input') :
            test_fovea = tf.placeholder(tf.float32, shape = [None, 64, 64, num_channels])
            test_embedding = tf.placeholder(tf.float32, shape = [None, 32])
            test_fish_prob = tf.placeholder(tf.float32, shape = [None, 1])
        with tf.name_scope('Network') :
            t_conv_output = convolutions(test_fovea)
            t_dense_input = tf.contrib.layers.flatten(t_conv_output)
            t_dense_output = dense_layers(t_dense_input, test_embedding, test_fish_prob, keep_prob = [1.0, 1.0, 1.0, 1.0])
        with tf.name_scope('Classifiers') :
            test_logits = tf.matmul(t_dense_output, W_clf) + b_clf
            test_predictions = tf.nn.sigmoid(test_logits)
