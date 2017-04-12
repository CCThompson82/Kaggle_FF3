"""This script is the FishFinder graph for the Kaggle Nature Conservancy Fishery
Competition.  It utilizes VGG-19 model as pre-trained weights during the
convolution steps"""


fish_finder = tf.Graph()

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
                W_fc4 = tf.Variable(tf.truncated_normal([fc_depth[2], fc_depth[3]], stddev = stddev ))
                b_fc4 = tf.Variable(tf.zeros([fc_depth[3]]))
                tf.summary.histogram('W_fc4', W_fc4)
                tf.summary.histogram('b_fc4', b_fc4)
        with tf.variable_scope('Classifiers') :
            with tf.name_scope('FishNoF') :
                W_clf = tf.Variable(tf.truncated_normal([fc_depth[3],1], stddev = stddev))
                b_clf = tf.Variable(tf.zeros([1]))
                tf.summary.histogram('W_clf', W_clf)
                tf.summary.histogram('b_clf', b_clf)
            with tf.name_scope('Scale') :
                W_box = tf.Variable(tf.truncated_normal([fc_depth[3], 3], stddev = stddev))
                b_box = tf.Variable(tf.zeros([3]))
                tf.summary.histogram('W_box', W_box)
                tf.summary.histogram('b_box', b_box)






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

    def dense_layers(data, keep_prob) :
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
        fc_layer = fc(fc_layer, W_fc4, b_fc4, keep_prob[3])
        return fc_layer




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            coarse_images = tf.placeholder(tf.float32, shape = [batch_size, coarse_dims[0], coarse_dims[1], num_channels])
            is_fish_labels = tf.placeholder(tf.float32, shape = [batch_size,1])
            box_targets = tf.placeholder(tf.float32, shape = [batch_size, 3])
            weights = tf.placeholder(tf.float32, shape = [batch_size, 1])


            learning_rate = tf.placeholder(tf.float32, shape = () )
            beta = tf.placeholder(tf.float32, shape = () )
            beta_regularizer = tf.placeholder(tf.float32, shape = ())
        with tf.name_scope('Network') :
            conv_output = convolutions(coarse_images)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, keep_prob = keep_prob)
        with tf.name_scope('Classifiers') :
            fishnof_logits = tf.matmul(dense_output, W_clf) + b_clf
            box_logits = tf.matmul(dense_output, W_box) + b_box
        with tf.name_scope('Backpropigation') :
            fishnof_xent = tf.nn.weighted_cross_entropy_with_logits(
                        logits = fishnof_logits, targets = is_fish_labels, pos_weight = beta)
            box_xent = tf.nn.sigmoid_cross_entropy_with_logits( logits = box_logits, targets = box_targets)
            cross_entropy_FishNoF = tf.reduce_mean(fishnof_xent)

            regularization_term = (tf.nn.l2_loss(W_fc4) +
                                   tf.nn.l2_loss(W_fc3) +
                                   tf.nn.l2_loss(W_fc2) +
                                   tf.nn.l2_loss(W_fc1)
                                  ) * beta_regularizer
            cross_entropy_box = tf.reduce_mean(tf.multiply(box_xent, weights))
            cost = cross_entropy_FishNoF + cross_entropy_box + regularization_term

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)


    with tf.name_scope('Validation') :
        with tf.name_scope('Input') :
            valid_set = tf.placeholder(tf.float32, shape = [200, coarse_dims[0], coarse_dims[1], num_channels])
            valid_labels = tf.placeholder(tf.float32, shape = [200,1])
            valid_box_targets = tf.placeholder(tf.float32, shape = [200, 3])
            valid_box_weights = tf.placeholder(tf.float32, shape = [200,1])
        with tf.name_scope('Network') :
            valid_conv_output = convolutions(valid_set)
            valid_dense_input = tf.contrib.layers.flatten(valid_conv_output)
            valid_dense_output = dense_layers(valid_dense_input, keep_prob = [1.0,1.0,1.0,1.0])
        with tf.name_scope('Prediction') :
            valid_logits = tf.matmul(valid_dense_output, W_clf) + b_clf
            valid_probs = tf.nn.sigmoid(valid_logits)
            valid_preds = tf.to_int32(tf.greater(valid_probs, 0.5))
            valid_acc = tf.reduce_mean(tf.cast(tf.equal( valid_preds, tf.to_int32(valid_labels)), tf.float32))
            valid_fish_xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = valid_logits, targets = tf.cast(valid_labels, tf.float32)))

            valid_box_logits = tf.matmul(valid_dense_output, W_box) + b_box
            valid_box_preds = tf.nn.sigmoid(valid_box_logits)
            valid_box_xent = tf.divide(
                                tf.reduce_mean(
                                    tf.multiply(
                                        tf.nn.weighted_cross_entropy_with_logits(
                                            logits = valid_box_logits,
                                            targets = valid_box_targets,
                                            pos_weight = valid_box_weights),
                                        valid_box_weights)),
                                tf.reduce_mean(valid_box_weights))


    with tf.name_scope('Summaries') :
        with tf.name_scope('Valid_Set') :
            tf.summary.histogram('Conv_output', valid_conv_output)
            tf.summary.histogram('FiNoF_Probability_distribution', valid_probs)
            tf.summary.scalar('Accuracy_FiNoF', valid_acc)
            tf.summary.scalar('FiNoF_CrossEntropy', valid_fish_xent)
            tf.summary.scalar('Box_CrossEntropy', valid_box_xent)
        with tf.name_scope('Parameters') :
            tf.summary.scalar('Beta_Reg_rate', beta_regularizer)
            tf.summary.scalar('LearningRate', learning_rate)
            tf.summary.scalar('beta', beta)
        with tf.name_scope('Train_Set') :
            tf.summary.scalar("Cost", cost)
            tf.summary.scalar('Regularization', regularization_term)
            tf.summary.scalar('Fish_Cross_entropy', cross_entropy_FishNoF)
            tf.summary.scalar('Box_Cross_entropy', tf.divide(cross_entropy_box, tf.reduce_mean(weights)))
        summaries = tf.summary.merge_all()


    with tf.name_scope('Prediction') :
        with tf.name_scope('Network') :
            coarse_images_for_prediction = tf.placeholder(dtype = tf.float32, shape = [None, coarse_dims[0], coarse_dims[1], num_channels])
            stack_conv_output = convolutions(coarse_images_for_prediction)
            stack_dense_input = tf.contrib.layers.flatten(stack_conv_output)
            stack_dense_output = dense_layers(stack_dense_input, keep_prob = [1.0,1.0,1.0,1.0])
        with tf.name_scope('Classifiers') :
            stack_FishNoF_preds = tf.nn.sigmoid(tf.matmul(stack_dense_output, W_clf) + b_clf)
            stack_box_preds = tf.nn.sigmoid(tf.matmul(stack_dense_output, W_box) + b_box)
