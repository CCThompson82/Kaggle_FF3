"""This is the session call associated with FishFinder_GRAPH.py"""


with tf.Session(graph = fishyfish) as session :

    # check for metadata dictionary
    if 'meta_dictionary.pickle' in os.listdir(md) and initiate_FishyFish != True:
        print("Loading FishyFish version {}".format(version_ID))
        with open(md+'/meta_dictionary.pickle', 'rb') as  handle :
            meta_dict = pickle.load(handle)
        print("Metadata dictionary loaded!")
        total_fovea = meta_dict.get(np.max([key for key in meta_dict])).get('examples_trained')
        epochs_completed = meta_dict.get(np.max([key for key in meta_dict])).get('Num_epochs')
        restorer = tf.train.Saver()
        print("Initializing restorer...")
        restorer.restore(session, tf.train.latest_checkpoint(md))
        print("Weights and biases retrieved!  Picking up at {} epochs completed : {} training images observed".format(epochs_completed, total_fovea))

    else :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        epochs_completed = 0
        total_fovea = 0
        meta_dict = {0 : {  'version_ID' : version_ID,
                            'examples_trained' : total_fovea,
                            'Num_epochs' : epochs_completed}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

    saver = tf.train.Saver()
    print("Checkpoint saver initialized!\n")
    saver.save(session, md+'/checkpoint', global_step = epochs_completed)

    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    print("\nTRAINING FishyFish {}...".format(version_ID))
    while open('FishyFish/stop.txt', 'r').read().strip() == 'False' :

        valid_embeddings, valid_FiNoF, valid_OH_labels, v_label_weights, valid_fov_stack, v_fovea_weights = (
                fd.prepare_FishyFish_batch(f_list = valid_fnames, embedding_df = embedding_df,
                                           annotated_fovea_directory = 'data/annotated_fovea_train/',
                                           predicted_fovea_directory = 'data/predicted_fovea_train/',
                                           annotated_boxes = annotated_boxes,
                                           box_preds = box_preds,
                                           label_df = labels,
                                           FiNoF_prob_series = FiNoF_prob,
                                           class_weight_dictionary = class_weight_dictionary,
                                           fov_weight_predicted = 0.2, fov_crop =64))


        epoch_keys = train_fnames.copy()

        start = datetime.now()
        while len(epoch_keys) > batch_size :
            # Choose random batch_size number of keys, removing from epoch_keys as selected
            batch_key_list = []
            for _ in range(batch_size) :
                batch_key_list.append( epoch_keys.pop(np.random.randint(0,len(epoch_keys))))

            em_arr, FiNoF_arr, labels_arr, lab_weights_arr, fov_arr, fov_weights_arr = (
                            fd.prepare_FishyFish_batch(f_list = batch_key_names, embedding_df = embedding_df,
                                       annotated_fovea_directory = 'data/annotated_fovea_train/',
                                       predicted_fovea_directory = 'data/predicted_fovea_train/',
                                       annotated_boxes = annotated_boxes,
                                       box_preds = box_preds,
                                       label_df = labels,
                                       FiNoF_prob_series = FiNoF_prob,
                                       class_weight_dictionary = class_weight_dictionary,
                                       fov_weight_predicted = 0.2, fov_crop =64))

            # Run a training step, output differs by whether summary is called or not.
            if len(epoch_keys) > batch_size :
                if (epochs_completed < 2) and ((total_fovea / batch_size) % 5 ) == 0:
                    feed_dict = {fovea : fov_arr,
                                 embedding : em_arr,
                                 fish_prob : FiNoF_arr,
                                 labels : labels_arr,
                                 fovea_weights : fov_weights_arr,
                                 label_weights : lab_weights_arr,
                                 learning_rate : float(open('FishyFish/learning_rate.txt', 'r').read().strip()),
                                 beta_regularizer : float(open('FishyFish/beta_reg.txt', 'r').read().strip()),
                                 gamma_fovea : float(open('FishyFish/fovea_coef.txt', 'r').read().strip()),
                                 gamma_label : float(open('FishyFish/label_coef.txt', 'r').read().strip()),
                                 val_fovea : valid_fov_stack,
                                 val_embedding : valid_embeddings,
                                 val_fish_prob : valid_FiNoF,
                                 val_labels : valid_OH_labels
                                 }

                    _, ce, summary_fetch = session.run([train_op, cost, summaries], feed_dict = feed_dict)
                    print("Batch Cost value:", ce)
                    writer.add_summary(summary_fetch, total_fovea)
                    total_fovea += batch_size

                else :
                    feed_dict = {fovea : fov_arr,
                                 embedding : em_arr,
                                 fish_prob : FiNoF_arr,
                                 labels : labels_arr,
                                 fovea_weights : fov_weights_arr,
                                 label_weights : lab_weights_arr,
                                 learning_rate : float(open('FishyFish/learning_rate.txt', 'r').read().strip()),
                                 beta_regularizer : float(open('FishyFish/beta_reg.txt', 'r').read().strip()),
                                 gamma_fovea : float(open('FishyFish/fovea_coef.txt', 'r').read().strip()),
                                 gamma_label : float(open('FishyFish/label_coef.txt', 'r').read().strip())
                                 }

                    _ = session.run([train_op], feed_dict = feed_dict)
                    total_fovea += batch_size
            else : # NOTE : This is the last batch before epoch ends ; summarize to tensorboard
                feed_dict = {fovea : fov_arr,
                             embedding : em_arr,
                             fish_prob : FiNoF_arr,
                             labels : labels_arr,
                             fovea_weights : fov_weights_arr,
                             label_weights : lab_weights_arr,
                             learning_rate : float(open('FishyFish/learning_rate.txt', 'r').read().strip()),
                             beta_regularizer : float(open('FishyFish/beta_reg.txt', 'r').read().strip()),
                             gamma_fovea : float(open('FishyFish/fovea_coef.txt', 'r').read().strip()),
                             gamma_label : float(open('FishyFish/label_coef.txt', 'r').read().strip()),
                             val_fovea : valid_fov_stack,
                             val_embedding : valid_embeddings,
                             val_fish_prob : valid_FiNoF,
                             val_labels : valid_OH_labels
                             }
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)
                total_fovea += batch_size
                writer.add_summary(summary_fetch, total_fovea)

        end = datetime.now()
        epoch_time = (end - start).total_seconds()
        epochs_completed += 1
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
        print("Epoch {} completed : {} coarse images observed in {} s ({} images/sec). Model Saved!".format(epochs_completed, total_fovea, epoch_time, len(label_dictionary)/epoch_time))
        meta_dict[epochs_completed] = {'Num_epochs' : epochs_completed,
                               'examples_trained' : total_fovea,
                               'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        
