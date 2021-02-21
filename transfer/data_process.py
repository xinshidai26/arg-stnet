import numpy as np
import sys

class DataGenerator(object):
    def __init__(self, data_path, dim_output, seq_length, threshold, temporal_type=""):
        self.data_path = data_path
        self.dim_output = dim_output
        self.seq_length = seq_length
        self.threshold = threshold
        self.cnnx_train = {}
        self.cnnx_test = {}
        self.y_train = {}
        self.y_test = {}
        self.cid_train = {}
        self.cid_test = {}
        self.seq_loc = {}
        self.cities = None
        self.cid_weight = None
        self.temporal_type = temporal_type
        if temporal_type == "period":
            self.p_cnnx_train = {}
            self.p_cnnx_test = {}

    def load_train_data(self, data_path, seq_length, time_end_type=0,
                        train_prop=0.8, period_seq_length=4):

        temporal_type = self.temporal_type
        volume = np.load(open('./raw_data/'+data_path, "rb"))['traffic']
        if time_end_type == 0:
            time_end = volume.shape[0]
        else:
            time_end = time_end_type
            volume = volume[-time_end:]
        data = volume / np.max(volume)
        cnn_features = []
        p_cnn_features = []
        labels = []
        valid_idx = []
        if temporal_type == "period":
            time_start = period_seq_length * 24
        else:
            time_start = seq_length
        for t in range(time_start, time_end):
            if temporal_type == "period":
                p_cnn_feature = [data[t - i * 24] for i in range(period_seq_length, 0, -1)]
                p_cnn_features.append(p_cnn_feature)
            cnn_feature = data[t-seq_length:t, :, :, :]
            cnn_features.append(cnn_feature)
            labels.append(data[t, :, :, :])
            max_volume = float(np.max(volume))
            norm_threshold = self.threshold / max_volume
            valid_value = data[t, :, :, :] >= norm_threshold
            if False in valid_value:
                valid_idx.append(False)
            else:
                valid_idx.append(True)
        cnn_features = np.array(cnn_features)
        if temporal_type == "period":
            p_cnn_features = np.array(p_cnn_features)
        labels = np.array(labels)
        if isinstance(train_prop, float):
            cnn_features = cnn_features[valid_idx]
            labels = labels[valid_idx]
            split_point = int(cnn_features.shape[0] * train_prop)
            self.cnnx_train[data_path] = cnn_features[:split_point]
            self.cnnx_test[data_path] = cnn_features[split_point:]
            self.y_train[data_path] = labels[:split_point]
            self.y_test[data_path] = labels[split_point:]
            if temporal_type == "period":
                p_cnn_features = p_cnn_features[valid_idx]
                self.p_cnnx_train[data_path] = p_cnn_features[:split_point]
                self.p_cnnx_test[data_path] = p_cnn_features[split_point:]
        elif isinstance(train_prop, int):
            split_point = train_prop
            train_valid_idx = valid_idx[:split_point]
            test_valid_idx = valid_idx[split_point:]
            self.cnnx_train[data_path] = cnn_features[:split_point][train_valid_idx]
            self.cnnx_test[data_path] = cnn_features[split_point:][test_valid_idx]
            self.y_train[data_path] = labels[:split_point][train_valid_idx]
            self.y_test[data_path] = labels[split_point:][test_valid_idx]
            if temporal_type == "period":
                self.p_cnnx_train[data_path] = p_cnn_features[:split_point][train_valid_idx]
                self.p_cnnx_test[data_path] = p_cnn_features[split_point:][test_valid_idx]
        print("train data shape:", self.cnnx_train[data_path].shape)
        print("train label shape:", self.y_train[data_path].shape)
        print("test data shape:", self.cnnx_test[data_path].shape)
        print("test label shape:", self.y_test[data_path].shape)
        if temporal_type == "period":
            print("train period shape:", self.p_cnnx_train[data_path].shape)
            print("test period shape:", self.p_cnnx_test[data_path].shape)
        return np.max(volume)

    def save_test_ground_truth(self, output_dir, data_path, test_data_num):
        if "bike" in data_path:
            np.savez(output_dir + "/output_bike_oracle", self.y_test[data_path][:test_data_num])
        else:
            np.savez(output_dir + "/output_oracle", self.y_test[data_path][:test_data_num])

    def generate(self, purpose, update_batch_size):
        data_path = self.data_path

        if purpose == "train":
            cnnx = self.cnnx_train[data_path]
            y = self.y_train
            if self.temporal_type == "period":
                p_cnnx = self.p_cnnx_train[data_path]
        else:
            cnnx = self.cnnx_test[data_path]
            y = self.y_test
            if self.temporal_type == "period":
                p_cnnx = self.p_cnnx_test[data_path]
        total_data_num = cnnx.shape[0]
        idx = np.random.choice(total_data_num, update_batch_size)
        inputs = cnnx[idx]
        outputs = np.array(y[data_path])[idx]
        if self.temporal_type == "period":
            p_inputs = p_cnnx[idx]
            return inputs, p_inputs, outputs
        else:
            return inputs, outputs

    def get_all_data(self, purpose):
        if purpose == "train":
            cnnx = self.cnnx_train[self.data_path]
            y = self.y_train[self.data_path]
            if self.temporal_type == "period":
                p_cnnx = self.p_cnnx_train[self.data_path]
        else:
            cnnx = self.cnnx_test[self.data_path]
            y = self.y_test[self.data_path]
            if self.temporal_type == "period":
                p_cnnx = self.p_cnnx_test[self.data_path]
        if self.temporal_type == "period":
            return np.array(cnnx), np.array(p_cnnx), np.array(y)
        else:
            return np.array(cnnx), np.array(y)
