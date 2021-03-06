"""This script is the FishFinder graph for the Kaggle Nature Conservancy Fishery
Competition.  It utilizes VGG-19 model as pre-trained weights during the
convolution steps"""


fish_finder = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(np.load(pretrained_path+'W_conv_1.npy'), trainable = True)
                b_conv1 = tf.Variable(np.load(pretrained_path+'b_conv_1.npy'), trainable = True)
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(np.load(pretrained_path+'W_conv_2.npy'), trainable = True)
                b_conv2 = tf.Variable(np.load(pretrained_path+'b_conv_2.npy'), trainable = True)
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(np.load(pretrained_path+'W_conv_3.npy'), trainable = True)
                b_conv3 = tf.Variable(np.load(pretrained_path+'b_conv_3.npy'), trainable = True)
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(np.load(pretrained_path+'W_conv_4.npy'), trainable = True)
                b_conv4 = tf.Variable(np.load(pretrained_path+'b_conv_4.npy'), trainable = True)
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(np.load(pretrained_path+'W_conv_5.npy'), trainable = True)
                b_conv5 = tf.Variable(np.load(pretrained_path+'b_conv_5.npy'), trainable = True)
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(np.load(pretrained_path+'W_conv_6.npy'), trainable = True)
                b_conv6 = tf.Variable(np.load(pretrained_path+'b_conv_6.npy'), trainable = True)
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
        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[3],1], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([1]))
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
            learning_rate = tf.placeholder(tf.float32, shape = () )
            beta = tf.placeholder(tf.float32, shape = () )
        with tf.name_scope('Network') :
            conv_output = convolutions(coarse_images)
            tf.summary.histogram('Conv_Output', conv_output)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, keep_prob = keep_prob)
        with tf.name_scope('Classifier') :
            logits = tf.matmul(dense_output, W_clf) #+ b_clf
        with tf.name_scope('Backpropigation') :
            xent = tf.nn.weighted_cross_entropy_with_logits(
                        logits = logits, targets = is_fish_labels, pos_weight = beta)
            cross_entropy = tf.reduce_mean(xent)
            cost = cross_entropy # + regularization or other cost amendment?

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)


    with tf.name_scope('Validation') :
        with tf.name_scope('Input') :
            valid_set = tf.constant(valid_coarse)
            valid_labels = tf.constant(valid_is_fish, dtype = tf.int32)
        with tf.name_scope('Network') :
            valid_conv_output = convolutions(valid_set)
            valid_dense_input = tf.contrib.layers.flatten(valid_conv_output)
            valid_dense_output = dense_layers(valid_dense_input, keep_prob = [1.0,1.0,1.0,1.0])
        with tf.name_scope('Prediction') :
            valid_probs = tf.nn.sigmoid(tf.matmul(valid_dense_output, W_clf)) #+ b_clf)
            tf.summary.histogram('Validation_Set_Probability', valid_probs)
            valid_preds = tf.to_int32(tf.greater(valid_probs, 0.5))
            valid_acc = tf.reduce_mean(tf.cast(tf.equal( valid_preds, valid_labels), tf.float32))

    with tf.name_scope('Summaries') :
        tf.summary.scalar('Weight_of_Fish', beta)
        tf.summary.scalar('Cross_entropy', cross_entropy)
        tf.summary.scalar('Learning_rate', learning_rate)
        tf.summary.scalar('Valid_Accuracy', valid_acc)
        summaries = tf.summary.merge_all()

    """
    with tf.name_scope('Prediction') :
        with tf.name_scope('Input') :
            img_stack = tf.placeholder(tf.float32, shape = [pred_batch, fov_size, fov_size, num_channels])
        with tf.name_scope('Network') :
            stack_conv_output, vgg_test_check = convolutions(img_stack)
            stack_dense_input = tf.contrib.layers.flatten(stack_conv_output)
            stack_dense_output = dense_layers(stack_dense_input, keep_prob = [1.0,1.0,1.0,1.0])
        with tf.name_scope('Classifier') :
            stack_prediction = tf.nn.softmax(tf.matmul(stack_dense_output, W_clf) + b_clf)
    """
