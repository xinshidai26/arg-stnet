import os
import argparse
import sys
import csv
import math
import numpy as np
import tensorflow as tf

from model import BasicModel
from data_process import DataGenerator


def func_save_result(data):
    if 'taxi' in data_path:
        csvfile = open('./new_result_taxi.csv', 'a+')
    else:
        csvfile = open('./new_result_bike.csv', 'a+')
    writer = csv.writer(csvfile)
    writer.writerow(data)
    csvfile.close()
    print('save already')

def eval_index(y, pred_y, epoch, save_mode=False, threshold=5):
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
    if save_mode:
        if temporal_type == 'period':
            list_result = [avg_mape_d, avg_mae_d, avg_rmse_d, avg_mae, avg_rmse, test_days,
                           s_city+"-period_loss_weight", epoch, meta_lr, l_value,
                           'period_loss_weight='+str(period_loss_weight)]
        else:
            list_result = [avg_mape_d, avg_mae_d, avg_rmse_d, avg_mae, avg_rmse, test_days,
                           s_city, epoch, meta_lr]
        func_save_result(list_result)

def test(model, data_generator, volume_max, sess, saver):
    epoch_value_set = 5
    train_inputs, train_labels = data_generator.get_all_data(purpose='train')
    test_inputs, test_labels = data_generator.get_all_data(purpose='test')
    train_data_size = train_inputs.shape[0]
    test_data_size = test_inputs.shape[0]
    train_batch_num = math.ceil(train_inputs.shape[0] / update_batch_size)
    test_batch_num = math.ceil(test_inputs.shape[0] / update_batch_size)
    for epoch in range(epochs):
        total_test_loss = []
        if epoch % epoch_value_set == 0:
            for i in range(test_batch_num):
                end_index = (i + 1) * update_batch_size
                if end_index <= test_inputs.shape[0]:
                    inputa = test_inputs[i * update_batch_size: (i + 1) * update_batch_size, :, :, :, :]
                    labela = test_labels[i * update_batch_size: (i + 1) * update_batch_size, :, :, :]
                else:
                    inputa = test_inputs[i * update_batch_size: test_data_size, :, :, :, :]
                    labela = test_labels[i * update_batch_size: test_data_size, :, :, :]
                    diff = end_index - test_data_size
                    diff_last = update_batch_size - diff
                    add_dat = test_inputs[0: diff, :, :, :, :]
                    add_label = test_labels[0: diff, :, :, :]
                    inputa = np.concatenate([inputa, add_dat], axis=0)
                    labela = np.concatenate([labela, add_label], axis=0)
                if temporal_type == "period":
                    p_inputa = np.zeros(shape=(update_batch_size, 4, inputa.shape[2], inputa.shape[3], 2))
                    feed_dict = {model.inputa: inputa, model.inputb: inputa,
                                 model.labela: labela, model.labelb: labela,
                                 model.p_inputa: p_inputa, model.p_inputb: p_inputa
                                 }
                    outputa, loss1 = sess.run([model.p_outputas, model.p_total_loss1], feed_dict)
                else:
                    feed_dict = {model.inputa: inputa, model.inputb: inputa,
                                 model.labela: labela, model.labelb: labela}
                    outputa, loss1 = sess.run([model.outputas, model.total_loss1], feed_dict)
                total_test_loss.append(loss1)
                if i == 0:
                    total_outputa = outputa
                elif end_index <= test_data_size:
                    total_outputa = np.concatenate([total_outputa, outputa], axis=0)
                else:
                    total_outputa = np.concatenate([total_outputa, outputa[:diff_last]], axis=0)
            actual_a = test_labels

            print("*" * 40)
            print("test:", epoch, np.sqrt(np.mean(total_test_loss)))
            eval_index(volume_max * actual_a.flatten(), volume_max * total_outputa.flatten(), epoch, save_mode=True)
            print("*"*20)
        total_train_loss = []
        for i in range(train_batch_num):
            end_index = (i + 1) * update_batch_size
            if end_index <= train_data_size:
                inputa = train_inputs[i * update_batch_size: (i + 1) * update_batch_size, :, :, :, :]
                labela = train_labels[i * update_batch_size: (i + 1) * update_batch_size, :, :, :]
            else:
                print("epoch:"+str(epoch)+" train_diff")
                inputa = train_inputs[i * update_batch_size: train_data_size, :, :, :, :]
                labela = train_labels[i * update_batch_size: train_data_size, :, :, :]
                diff = end_index - train_data_size
                diff_last = update_batch_size - diff
                add_dat = train_inputs[0: diff, :, :, :, :]
                add_label = train_labels[0: diff, :, :, :]
                inputa = np.concatenate([inputa, add_dat], axis=0)
                labela = np.concatenate([labela, add_label], axis=0)
            if temporal_type == "period":
                p_inputa = np.zeros(shape=(update_batch_size, 4, inputa.shape[2], inputa.shape[3], 2))
                feed_dict = {model.inputa: inputa, model.inputb: inputa,
                             model.labela: labela, model.labelb: labela,
                             model.p_inputa: p_inputa, model.p_inputb: p_inputa
                             }
                sess.run([model.p_train_op], feed_dict)
                outputa, loss1 = sess.run([model.p_outputas, model.p_total_loss1], feed_dict)
            else:
                feed_dict = {model.inputa: inputa, model.inputb: inputa,
                             model.labela: labela, model.labelb: labela}
                sess.run([model.pretrain_op], feed_dict)
                outputa, loss1 = sess.run([model.outputas, model.total_loss1], feed_dict)

            total_train_loss.append(loss1)

            if i == 0:
                total_outputa = outputa
            elif end_index <= train_data_size:
                total_outputa = np.concatenate([total_outputa, outputa], axis=0)
            else:
                total_outputa = np.concatenate([total_outputa, outputa[:diff_last]], axis=0)
        actual_a = train_labels
        if epoch % epoch_value_set == 0:
            print("train:", epoch, np.sqrt(np.mean(total_train_loss)))
            eval_index(volume_max * actual_a.flatten(), volume_max * total_outputa.flatten(), epoch, save_mode=False)
    print(volume_max)

def main():
    tf.set_random_seed(1234)
    time_end_type = 31*24
    print("target" + "*"*20)
    data_generator = DataGenerator(data_path=data_path,
                                   dim_output=dim_output,
                                   seq_length=seq_length,
                                   threshold=threshold)
    if dim_output == 2:
        volume_max = data_generator.load_train_data(data_path, seq_length, time_end_type=time_end_type,
                                                    train_prop=int(test_days*24))
    else:
        print('dim_output error')
        sys.exit()
    train_inputs, train_labels = data_generator.get_all_data(purpose='train')
    input_height = train_labels.shape[1]
    input_width = train_labels.shape[2]
    input_size_list = [input_height, input_width]
    model = BasicModel(input_size=input_size_list, dim_output=dim_output, seq_length=seq_length,
                       filter_num=32, meta_lr=meta_lr, update_batch_size=update_batch_size, clstm_bidden_num=64,
                       temporal_type=temporal_type)
    model.construct_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    if len(save_dir) > 0:
        model_file = save_dir + "/" + s_city + temporal_type + "/" + test_model_name
        print(model_file)
        saver.restore(sess, model_file)
        print("Testing:", model_file, "with %d days data" % test_days)
    else:
        print("Target data only", "with %d days data" % test_days)
    print("Training:")
    test(model, data_generator, volume_max, sess, saver)


if __name__ == "__main__":
    # python transfer/test.py --data_path=bike/bike_chicago.npz --s_city= --save_dir= --cities=none --test_days=1
    # python transfer/test.py --data_path=bike/bike_chicago.npz --test_model=model_5000 --s_city=att_reptile_bike --cities=chicago --test_days=1 --temporal_type=period
    # python transfer/test.py --data_path=taxi/nyc.npz --cities=nyc --save_dir= --test_days=3
    # python transfer/test.py --data_path=taxi/nyc.npz --test_model=model_5000 --s_city=att_reptile_taxi --cities=nyc --temporal_type=period --test_days=3
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_path')
    parser.add_argument('--cities', type=str, default='nyc')
    parser.add_argument('--save_dir', type=str, default='./transfer/models')
    parser.add_argument('--output_dir', type=str, default='./transfer/outputs')
    parser.add_argument('--temporal_type', type=str, default='')
    parser.add_argument('--test_model', type=str)
    parser.add_argument('--s_city', type=str)
    parser.add_argument('--test_days', type=int)
    parser.add_argument('--update_batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--meta_lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--period_loss_weight', type=float, default=0.5)
    parser.add_argument('--l_value', type=float, default=0.5)
    parser.add_argument('--gpu_id', type=str, default="0")
    dim_output = 2
    seq_length = 8
    args = parser.parse_args()
    data_path = args.data_path
    save_dir = args.save_dir
    test_days = args.test_days
    output_dir = args.output_dir
    cities = args.cities
    s_city = args.s_city
    l_value = args.l_value
    update_batch_size = args.update_batch_size
    threshold = args.threshold
    meta_lr = args.meta_lr
    epochs = args.epochs
    test_model_name = args.test_model
    temporal_type = args.temporal_type
    period_loss_weight = args.period_loss_weight
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main()