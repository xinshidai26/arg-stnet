import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import time
from model import Att_ReptileModel
from att_data_process import DataGenerator

def eval_index(y, pred_y, threshold=5):
    avg_rmse = np.sqrt(np.mean(np.square(y-pred_y)))
    avg_mae = np.mean(np.abs(y - pred_y))
    mape_mask = y > threshold
    avg_rmse_d = np.sqrt(np.mean(np.square(y[mape_mask] - pred_y[mape_mask])))
    avg_mae_d = np.mean(np.abs(y[mape_mask] - pred_y[mape_mask]))
    avg_mape_d = np.mean(np.abs(y[mape_mask] - pred_y[mape_mask]) / y[mape_mask])
    print("avg_mape_d:,", avg_mape_d)
    print("avg_mae_d:,", avg_mae_d)
    print("avg_rmse_d:,", avg_rmse_d)
    print("avg_mae:,", avg_mae)
    print("avg_rmse:,", avg_rmse)

def train(model, data_generator, t_data_generator, sess, saver):
    model.model_state()
    time_1 = time.time()
    for epoch in range(iterations):
        if epoch % 100 == 0:
            input_list = {}
            if temporal_type == "period":
                batch_x, p_batch_x, batch_y = data_generator.generate(purpose='train',
                                                                      update_batch_size=update_batch_size)
                input_list['inputa'] = batch_x
                input_list['p_inputa'] = p_batch_x
                input_list['labela'] = batch_y
                feed_dict = {model.input1_a: input_list['inputa'], model.p_input1_a: input_list['p_inputa'],
                             model.label1_a: input_list['labela']}
                res = sess.run([model.total_rmse_list, model.p_total_loss_rmse_list, model.period_rmse_list], feed_dict)
            else:
                batch_x, batch_y = data_generator.generate(purpose='train',
                                                           update_batch_size=update_batch_size)
                input_list['inputa'] = batch_x
                input_list['labela'] = batch_y
                feed_dict = {model.input1_a: input_list['inputa'], model.label1_a: input_list['labela']}
                res = sess.run([model.total_rmse_list], feed_dict)
            print(epoch, "train", res)
            model_file = save_dir + "/" + save_name + temporal_type + "/model_" + str(epoch)
            saver.save(sess, model_file)
            if temporal_type == "period":
                batch_x, p_batch_x, batch_y = data_generator.generate(purpose='test',
                                                                      update_batch_size=update_batch_size)
                input_list['inputa'] = batch_x
                input_list['p_inputa'] = p_batch_x
                input_list['labela'] = batch_y
                feed_dict = {model.input1_a: input_list['inputa'], model.p_input1_a: input_list['p_inputa'],
                             model.label1_a: input_list['labela']}
                res = sess.run([model.total_rmse_list, model.p_total_loss_rmse_list, model.period_rmse_list], feed_dict)
            else:
                batch_x, batch_y = data_generator.generate(purpose='test',
                                                           update_batch_size=update_batch_size)
                input_list['inputa'] = batch_x
                input_list['labela'] = batch_y
                feed_dict = {model.input1_a: input_list['inputa'], model.label1_a: input_list['labela']}
                res = sess.run([model.total_rmse_list], feed_dict)
            print("test", res)
            time_2 = time.time()
            print("time_interval_"+str(epoch)+":", time_2-time_1)
            time_1 = time.time()
        else:
            model.train_step(data_generator, t_data_generator)

def main():
    tf.set_random_seed(1234)
    data_generator = DataGenerator(cities_data_path=data_path,
                                   dim_output=dim_output,
                                   seq_length=seq_length,
                                   threshold=threshold,
                                   temporal_type=temporal_type)
    if dim_output == 2:
        volume_list = data_generator.load_train_data(data_path, seq_length, train_prop=0.8)
    else:
        print('dim_output error')
        sys.exit()
    if temporal_type == "period":
        train_inputs, p_train_inputs, train_labels = data_generator.get_all_data(purpose='train')
    else:
        train_inputs, train_labels = data_generator.get_all_data(purpose='train')
    input_size = []

    for i in range(len(train_labels)):
        input_height = train_labels[i].shape[2]
        input_width = train_labels[i].shape[3]
        input_size.append(input_height)
        input_size.append(input_width)
    t_data_generator = DataGenerator(cities_data_path=t_data_path,
                                     dim_output=dim_output,
                                     seq_length=seq_length,
                                     threshold=threshold,
                                     temporal_type='')
    time_end_type = 31 * 24
    if dim_output == 2:
        t_data_generator.load_train_data(t_data_path, seq_length, time_end_type=time_end_type,
                                         train_prop=int(test_days*24))
    else:
        print('dim_output error')
    train_inputs, train_labels = t_data_generator.get_all_data(purpose='train')
    t_input_size = []
    for i in range(len(train_labels)):
        input_height = train_labels[i].shape[2]
        input_width = train_labels[i].shape[3]
        t_input_size.append(input_height)
        t_input_size.append(input_width)
    sess = tf.InteractiveSession()
    model = Att_ReptileModel(input_size=input_size, t_input_size=t_input_size, dim_output=dim_output, seq_length=seq_length, session=sess,
                             test_num_updates=test_num_updates, meta_batch_size=len(data_path), filter_num=32,
                             meta_lr=meta_lr, update_lr=update_lr, update_batch_size=update_batch_size,
                             clstm_bidden_num=64, temporal_type=temporal_type,
                             period_loss_weight=period_loss_weight, l_value=l_value)
    model.construct_model()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    print("Training:")
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    train(model, data_generator, t_data_generator, sess, saver)

if __name__ == "__main__":
    # python transfer/att_rep_train.py  --iterations=5001 --meta_lr=1e-0 --temporal_type=period --period_loss_weight=0.1
    # python transfer/att_rep_train.py --data_path=taxi/porto.npz,taxi/dc.npz --save_name=att_reptile_taxi --t_data_path=taxi/nyc.npz --update_batch_size=16 --temporal_type=period --meta_lr=1e-1
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='bike/bike_nyc.npz,bike/bike_dc.npz')
    parser.add_argument('--t_data_path', type=str, default='bike/bike_chicago.npz')
    parser.add_argument('--save_name', type=str, default='att_reptile_bike')
    parser.add_argument('--save_dir', type=str, default='./transfer/models')
    parser.add_argument('--update_batch_size', type=int, default=16)
    parser.add_argument('--test_num_updates', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--period_loss_weight', type=float, default=0.5)
    parser.add_argument('--l_value', type=float, default=0.1)
    parser.add_argument('--test_days', type=int, default=1)
    parser.add_argument('--meta_lr', type=float, default=1e-0)
    parser.add_argument('--update_lr', type=float, default=1e-3)
    parser.add_argument('--iterations', type=int, default=5001)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--temporal_type', type=str, default='')
    dim_output = 2
    seq_length = 8
    args = parser.parse_args()
    data_path = args.data_path.split(',')
    t_data_path = args.t_data_path.split(',')
    save_dir = args.save_dir
    save_name = args.save_name
    update_batch_size = args.update_batch_size
    test_num_updates = args.test_num_updates
    update_lr = args.update_lr
    threshold = args.threshold
    test_days = args.test_days
    meta_lr = args.meta_lr
    l_value = args.l_value
    iterations = args.iterations
    temporal_type = args.temporal_type
    period_loss_weight = args.period_loss_weight
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main()