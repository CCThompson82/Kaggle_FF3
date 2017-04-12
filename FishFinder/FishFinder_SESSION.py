"""This is the session call associated with FishFinder_GRAPH.py"""


with tf.Session(graph = fish_finder) as session :

    # check for metadata dictionary
    if 'meta_dictionary.pickle' in os.listdir(md) and initiate_FishFinder != True:
        print("Loading FishFinder_MT version {}".format(version_ID))
        with open(md+'/meta_dictionary.pickle', 'rb') as  handle :
            meta_dict = pickle.load(handle)
        print("Metadata dictionary loaded!")
        total_fovea = meta_dict.get(np.max([key for key in meta_dict])).get('fovea_trained')
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
                            'fovea_trained' : total_fovea,
                            'Num_epochs' : epochs_completed}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)
        with open('prediction_dictionary.pickle', 'wb') as fprd :
            pickle.dump({'version' : version_ID}, fprd)

    saver = tf.train.Saver()
    print("Checkpoint saver initialized!\n")
    saver.save(session, md+'/checkpoint', global_step = epochs_completed)

    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    print("\nTRAINING FishFinder_MT {}...".format(version_ID))
    while open('FishFinder/stop.txt', 'r').read().strip() == 'False' :
        # NOTE : label_dictionary will be
        with open('label_dictionary.pickle', 'rb') as handle :
            label_dictionary = pickle.load(handle)

        master = fd.generate_filenames_list()
        valid_fnames = []
        train_fnames = []

        for key in master :
            if label_dictionary.get(key).get('valid_set') == True :
                valid_fnames.append(key)
            else :
                train_fnames.append(key)

        epoch_keys = train_fnames.copy()

        valid_coarse, valid_is_fish, valid_box, valid_weights = fd.bundle_mt(valid_fnames, label_dictionary = label_dictionary)
        valid_coarse = valid_coarse.astype(np.float32)
        for i in range(valid_coarse.shape[0]) :
            valid_coarse[i,:, :, :] = fd.process_fovea(valid_coarse[i,:,:,:], pixel_norm = 'centre', mutation = False)

        start = datetime.now()
        while len(epoch_keys) > batch_size :
            # Choose random batch_size number of keys, removing from epoch_keys as selected
            batch_key_list = []
            for _ in range(batch_size) :
                batch_key_list.append( epoch_keys.pop(np.random.randint(0,len(epoch_keys))))

            batch_arr, is_fish_vector, box_arr, weights_vector = fd.bundle_mt(batch_key_list, label_dictionary = label_dictionary, fov_dim = fov_dim)
            for ix in range(batch_size) :
                batch_arr[ix, :, :, :] = fd.process_fovea(batch_arr[ix,:,:,:], pixel_norm = 'centre', mutation = False)  # TODO : Refactor so fn handles batch arrays and get rid of this loop

            # Run a training step, output differs by whether summary is called or not.
            if len(epoch_keys) > batch_size :
                if (epochs_completed < 2) and ((total_fovea / batch_size) % 5 ) == 0:
                    feed_dict = {coarse_images : batch_arr,
                                 is_fish_labels : is_fish_vector,
                                 box_targets : box_arr,
                                 weights : weights_vector,
                                 valid_set : valid_coarse,
                                 valid_labels : valid_is_fish,
                                 valid_box_targets : valid_box,
                                 valid_box_weights : valid_weights,
                                 learning_rate : float(open('FishFinder/learning_rate.txt', 'r').read().strip()),
                                 beta : float(open('FishFinder/beta_rate.txt', 'r').read().strip()),
                                 beta_regularizer : float(open('FishFinder/beta_reg.txt', 'r').read().strip())
                                 }

                    _, ce, summary_fetch = session.run([train_op, cost, summaries], feed_dict = feed_dict)
                    print("Batch Cost value:", ce)
                    writer.add_summary(summary_fetch, total_fovea)
                    total_fovea += batch_size

                else :
                    feed_dict = {coarse_images : batch_arr,
                                 is_fish_labels : is_fish_vector,
                                 box_targets : box_arr,
                                 weights : weights_vector,
                                 learning_rate : float(open('FishFinder/learning_rate.txt', 'r').read().strip()),
                                 beta : float(open('FishFinder/beta_rate.txt', 'r').read().strip()),
                                 beta_regularizer : float(open('FishFinder/beta_reg.txt', 'r').read().strip())
                                 }

                    _ = session.run([train_op], feed_dict = feed_dict)
                    total_fovea += batch_size
            else : # NOTE : This is the last batch before epoch ends ; summarize to tensorboard
                feed_dict = {coarse_images : batch_arr,
                             is_fish_labels : is_fish_vector,
                             box_targets : box_arr,
                             weights : weights_vector,
                             valid_set : valid_coarse,
                             valid_labels : valid_is_fish,
                             valid_box_targets : valid_box,
                             valid_box_weights : valid_weights,
                             learning_rate : float(open('FishFinder/learning_rate.txt', 'r').read().strip()),
                             beta : float(open('FishFinder/beta_rate.txt', 'r').read().strip()),
                             beta_regularizer : float(open('FishFinder/beta_reg.txt', 'r').read().strip())
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
                               'fovea_trained' : total_fovea,
                               'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        # if counter == 0 : retrieve predictions
        if counter == 0 :
            print("Running Predictor on Training Examples...")

            #retrieve most up to date label_dictionary
            with open('label_dictionary.pickle', 'rb') as handle :
                label_dictionary = pickle.load(handle)

            prediction_keys = master.copy()
            embedding_arr = np.zeros([len(label_dictionary), 32])
            cursor = 0
            while len(prediction_keys) > batch_size :
                batch_key_list = []
                for _ in range(batch_size) :
                    batch_key_list.append( prediction_keys.pop(0))

                batch_arr, is_fish_vector, box_arr, weights_vector = fd.bundle_mt(batch_key_list, label_dictionary = label_dictionary, fov_dim = fov_dim)
                feed_dict = {coarse_images_for_prediction : batch_arr}

                FiNoF_Probability, Box_Predictions, coarse_embedding = session.run([stack_FishNoF_preds, stack_box_preds, stack_dense_output], feed_dict = feed_dict)

                for i, key in enumerate(batch_key_list) :
                    label_dictionary[key].update({'FiNoF' : FiNoF_Probability[i],
                                                  'box_preds' : Box_Predictions[i,:]})
                    embedding_arr[cursor, :] = coarse_embedding[i, :]
                    if (cursor % 256) == 0 :
                        print("{} images embedded".format(cursor))
                        print("Length of prediction_keys : {}".format(len(prediction_keys)))
                    cursor += 1


            # last run with leftovers
            batch_arr, is_fish_vector, box_arr, weights_vector = fd.bundle_mt(prediction_keys, label_dictionary = label_dictionary, fov_dim = fov_dim)
            feed_dict = {coarse_images_for_prediction : batch_arr}

            FiNoF_Probability, Box_Predictions, coarse_embedding = session.run([stack_FishNoF_preds, stack_box_preds, stack_dense_output], feed_dict = feed_dict)

            for i, key in enumerate(prediction_keys) :
                label_dictionary[key].update({'FiNoF' : FiNoF_Probability[i],
                                              'box_preds' : Box_Predictions[i,:]})
                embedding_arr[cursor, :] = coarse_embedding[i, :]
                cursor += 1



            with open('label_dictionary.pickle', 'wb') as fld :
                pickle.dump(label_dictionary, fld)

            embedding_df = pd.DataFrame(embedding_arr, index = master.copy())
            embedding_df.to_pickle('embedding_dataframe.pickle')

            counter = predict_every_z
            print("Prediction of training examples finished and saved to working directory")
        else :
            counter -= 1
