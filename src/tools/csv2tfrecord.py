import random

import numpy as np
import tfrecord
import torch
from dataset import RecIterableDataset
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)


def convert_csv_to_tfrecord(csv_path, tf_path, config, val_ratio=-1, val_tf_path=None, seed=0):
    ds = RecIterableDataset([csv_path], config)
    writer_train = tfrecord.TFRecordWriter(tf_path)
    writer_val = tfrecord.TFRecordWriter(val_tf_path)
    random.seed(seed)
    for i, sample in enumerate(ds):
        click, conversion, features = sample
        # features = np.concatenate(features)
        sample_to_tfrecord = {
            "click": (click, "int"),
            "conversion": (conversion, "int"),
            "features": (features, "int"),
        }
        if random.random() >= val_ratio:
            writer_train.write(sample_to_tfrecord)
        else:
            writer_val.write(sample_to_tfrecord)

    writer_train.close()
    writer_val.close()


if __name__ == "__main__":
    train_csv_path = "../data2/train.csv"
    train_tf_path = "../data/train.tfrecord"
    train_index_path = "../data/train.tfindex"
    val_tf_path = "../data/val.tfrecord"
    val_index_path = "../data/val.tfindex"
    test_csv_path = "../data2/test.csv"
    test_tf_path = "../data/test.tfrecord"
    test_index_path = "../data/test.tfindex"

    # using_feature_ids = [str(i) for i in range(333)]
    using_feature_ids = ['322', '320', '329', '328', '31', '319', '321', '302', '324', '323', '325', '32', '301', '326', '300', '303', '307', '33', '310', '311', '309', '312', '308', '299', '305', '304', '306', '327', '290', '291', '298', '313', '289', '294', '297', '296', '293', '100', '318', '314', '287', '288', '315', '276', '273', '274', '275', '279', '277', '292', '272', '316', '278', '271', '280', '317', '282', '281', '270', '283', '284', '268', '267', '269', '286', '265', '261', '266', '285', '262', '264', '170', '168', '142', '169', '171', '161', '167', '162', '151', '263', '144', '164', '143', '153', '172', '163', '155', '156', '159', '160', '145', '152', '157', '166', '245', '165', '150', '250', '260']
    # using_feature_ids = [str(i) for i in range(330)]
    # print('295' in using_feature_ids)
    # using_feature_ids.pop(295)
    # print('295' in using_feature_ids)

    config = dict(
        single_feature_len=3,
        using_feature_ids=using_feature_ids,
        feature_names=[i for i in range(len(using_feature_ids))],
        features_num=len(using_feature_ids),
    )
    print('start')
    # convert_csv_to_tfrecord(
    #     train_csv_path, train_tf_path, config, val_ratio=0.01, val_tf_path=val_tf_path, seed=2020
    # )
    
    # print('finish train')
    convert_csv_to_tfrecord(
        test_csv_path, test_tf_path, config, val_ratio=-1, val_tf_path="../data/tmp.tfrecord"
    )
    print('finish test')
    # convert_csv_to_tfrecord(
    #     train_csv_path,
    #     "../data/tmp.tfrecord",
    #     config,
    #     val_ratio=0.01,
    #     val_tf_path="../data/tmp_val.tfrecord",
    #     seed=20
    # )

    # description = {'click':'int', 'conversion':'int', 'features':'int'}
    # train_dataset = TFRecordDataset(test_tf_path,
    #                                 index_path=test_index_path,
    #                                 shuffle_queue_size=1024,
    #                                 description=description,
    #                                 )
    # train_dataloder = DataLoader(train_dataset, batch_size=10000)
    # conversion_num = 0
    # click_sum = 0
    # for i, batch in enumerate(train_dataloder):
    #     click, conversion, features = batch['click'].squeeze(1), batch['conversion'].squeeze(1), batch['features']
    #     features = features.reshape(len(features), config['features_num'], config['single_feature_len'])
    #     click_sum += click.sum()
    #     conversion_num += conversion.sum()

    #     if i == 10:
    #         print('stop')
    #         break
