# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import numpy as np
import config as config
config = config.config

import utils
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import network
import data_manager
from dataset_loader import ImageDataset

from utils import Hamming_Score as hamming_accuracy

def train_teacher():
    """
    This function trains a teacher (teacher id) among an ensemble of nb_teachers
    models for the dataset specified.
    :param dataset: string corresponding to dataset (svhn, cifar10)
    :param nb_teachers: total number of teachers in the ensemble
    :param teacher_id: id of the teacher being trained
    :return: True if everything went well
    """
    # If working directories do not exist, create them
    #assert utils.mkdir_if_missing(config.data_dir)
    #assert utils.mkdir_if_misshing(config.train_dir)
    print("Initializing dataset {}".format(config.dataset))

    dataset = data_manager.init_img_dataset(
        root=config.data_dir, name=config.dataset,
    )

    # Load the dataset


    for i in range(0,config.nb_teachers):
        # Retrieve subset of data for this teacher

        if config.dataset == 'celeba':
            data, labels = dataset._data_partition(config.nb_teachers,i)

       
        print("Length of training data: " + str(len(data)))

        # Define teacher checkpoint filename and full path
        print('data.shape for each teacher')


        dir_path = os.path.join(config.save_model,'pate_'+config.dataset+str(config.nb_teachers))
        utils.mkdir_if_missing(dir_path)
        #filename = os.path.join(dir_path, str(config.nb_teachers) + '_teachers_' + str(i) + '_resnet.checkpoint.pth.tar')
        filename = os.path.join(dir_path, str(config.nb_teachers) + '_teachers_' + str(i) + config.arch+'.checkpoint.pth.tar')
        print('save_path for teacher{}  is {}'.format(i,filename))

        network.train_each_teacher(config.teacher_epoch,data, labels, dataset.test_data, dataset.test_label, filename)


    return True




def main(argv=None):  # pylint: disable=unused-argument
    # Make a call to train_teachers with values specified in flags
    train_teacher()
main()
