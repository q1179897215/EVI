import itertools
from collections import defaultdict

import lightning as L
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import tfrecord
import torch
import torch.nn as nn
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset


class XDataset(Dataset):
    """Load csv data with feature name ad first row."""

    def __init__(self, datafile, debug=False):
        super().__init__()
        self.feature_names = []
        self.debug = debug
        self.datafile = datafile
        self.data = []
        self._load_data()

    def _load_data(self):
        print(f"start load data from: {self.datafile}")
        if self.debug:
            df = pd.read_hdf(self.datafile, key="df", stop=200001)
        else:
            df = pd.read_hdf(self.datafile, key="df")
        df = df.astype(int)
        self.feature_names = df.columns[2:]
        self.data = df.values
        print(
            f"load data from {self.datafile} finished, there is {len(self.data)} samples"
        )

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        click = line[0]
        conversion = line[1]
        features = dict(zip(self.feature_names, line[2:]))
        return click, conversion, features


class AliExpressDataset(Dataset):
    def __init__(self, dataset_path, debug=False):
        if debug:
            df = pd.read_hdf(dataset_path, key="df", stop=200001)
        else:
            df = pd.read_hdf(dataset_path, key="df")
        data = df.to_numpy()[:, 1:]
        self.categorical_data = data[:, :16].astype(np.int)
        self.numerical_data = data[:, 16:-2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return (
            self.labels[index][0],
            self.labels[index][1],
            (self.categorical_data[index], self.numerical_data[index]),
        )


class RecIterableDataset(IterableDataset):
    def __init__(self, file_list, config):
        self.config = config
        self.file_list = file_list
        self.init()

    def init(self):
        all_field_id = self.config.get("using_feature_ids", [str(i) for i in range(333)])
        self.all_field_id_dict = defaultdict(int)
        self.max_len = self.config.get("single_feature_len", 3)
        for i, field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False, i]
        self.padding = 10614789

    def preprocess(self, features):
        ctr = np.int64(features[1])
        ctcvr = np.int64(features[2])
        output = [(field_id, []) for field_id in self.all_field_id_dict]
        output_list = []
        for elem in features[3:]:
            field_id, feat_id = elem.strip().split(":")
            if field_id not in self.all_field_id_dict:
                continue
            self.all_field_id_dict[field_id][0] = True
            index = self.all_field_id_dict[field_id][1]
            output[index][1].append(int(feat_id))
        for field_id, (visited, index) in self.all_field_id_dict.items():
            self.all_field_id_dict[field_id][0] = False
            if len(output[index][1]) > self.max_len:
                output_list.append(output[index][1][: self.max_len])
            else:
                output_list.append(
                    output[index][1] + [self.padding] * (self.max_len - len(output[index][1]))
                )
        output_list = np.array(output_list, dtype="int64")
        # output_list = [i for i in output_list]
        # concatenate the outputlist which is a list
        output_list = np.concatenate(output_list, axis=0, dtype="int64")

        return ctr, ctcvr, output_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            for file in self.file_list:
                with open(file) as rf:
                    for l in rf:
                        features = l.strip().split(",")
                        yield self.preprocess(features)
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            for file in self.file_list:
                with open(file) as rf:
                    # Split the file into chunks based on the number of workers
                    chunks = itertools.islice(rf, worker_id, None, num_workers)
                    for l in chunks:
                        features = l.strip().split(",")
                        yield self.preprocess(features)


class CCPLightningDataModule(L.LightningDataModule):
    def __init__(self, batch_size=8000, num_workers=8, debug=False, batch_type="ccp"):
        super().__init__()
        self.train_dataset = XDataset("../data/ctr_cvr.train.h5", debug=debug)
        self.val_dataset = XDataset("../data/ctr_cvr.dev.h5", debug=debug)
        self.test_dataset = XDataset("../data/ctr_cvr.test.h5", debug=debug)
        self.field_dims = 0
        self.numerical_num = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class INLightningDataModule(L.LightningDataModule):
    def __init__(
        self, batch_size=8000, num_workers=8, debug=False, shuffle_queue_size=512, batch_type="in"
    ):
        self.shuffle_queue_size = shuffle_queue_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.batch_type = batch_type
        super().__init__()
        train_data_path = "../data/train.tfrecord"
        train_data_path_index = "../data/train.tfindex"
        val_data_path = "../data/val.tfrecord"
        val_data_path_index = "../data/val.tfindex"
        test_data_path = "../data/test.tfrecord"
        test_data_path_index = "../data/test.tfindex"
        description = {"features": "int", "click": "int", "conversion": "int"}

        self.train_dataset = TFRecordDataset(
            train_data_path,
            index_path=train_data_path_index,
            description=description,
            shuffle_queue_size=self.shuffle_queue_size,
        )
        self.val_dataset = TFRecordDataset(
            val_data_path,
            index_path=val_data_path_index,
            description=description,
            shuffle_queue_size=self.shuffle_queue_size,
        )
        self.test_dataset = TFRecordDataset(
            test_data_path,
            index_path=test_data_path_index,
            description=description,
            shuffle_queue_size=self.shuffle_queue_size,
        )

        self.field_dims = 0
        self.numerical_num = 0

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class ExpressLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size=8000,
        num_workers=8,
        debug=False,
        train_data_path="../data/AliExpress_FR/train.hdf5",
        test_data_path="../data/AliExpress_FR/test.hdf5",
        batch_type="fr",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.train_dataset = AliExpressDataset(train_data_path, debug=debug)
        self.test_dataset = AliExpressDataset(test_data_path, debug=debug)
        self.field_dims = self.train_dataset.field_dims
        self.numerical_num = self.train_dataset.numerical_num

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    # train_csv_path = '../data/train.csv'
    # train_tf_path = '../data/train.tfrecord'
    # train_index_path = '../data/train.tfindex'
    # val_tf_path = '../data/val.tfrecord'
    # val_index_path = '../data/val.tfindex'
    # test_csv_path = '../data/test.csv'
    # test_tf_path = '../data/test.tfrecord'
    # test_index_path = '../data/test.tfindex'
    # using_feature_ids = ['322', '320', '329', '328', '331', '319', '321', '302', '324', '323', '325', '332', '301', '326', '300', '303', '307', '330', '310', '311', '309', '312', '308', '299', '305', '304', '306', '327', '290', '291', '298', '313', '289', '294', '297', '296', '293', '295', '318', '314', '287', '288', '315', '276', '273', '274', '275', '279', '277', '292', '272', '316', '278', '271', '280', '317', '282', '281', '270', '283', '284', '268', '267', '269', '286', '265', '261', '266', '285', '262', '264', '170', '168', '142', '169', '171', '161', '167', '162', '151', '263', '144', '164', '143', '153', '172', '163', '155', '156', '159', '160', '145', '152', '157', '166', '245', '165', '150', '250', '260']
    # config = dict(
    #     single_feature_len=3,
    #     using_feature_ids = using_feature_ids,
    #     feature_names = [i for i in range(len(using_feature_ids))],
    #     features_num = len(using_feature_ids)
    #     )
    # train_dataset = RecIterableDataset([train_csv_path], config)
    # train_dl = DataLoader(train_dataset, batch_size=2000)
    # for i, sample in enumerate(train_dl):
    #     click, conversion, features = sample
    #     break
    root = "../data/AliExpress_US/"
    debug = True
    train_dataset = AliExpressDataset(root + "train.hdf5", debug=debug)
    val_dataset = None
    test_dataset = AliExpressDataset(root + "test.hdf5", debug=debug)
    print(train_dataset[5])
