#! /usr/bin/python3 

"""

@authors: Long Wang (long@subtlemedical.com)
Copyright (c) Subtle Medical, Inc.
"""

import h5py
import os
import numpy as np
import keras
import logging
import time
import threading 
import random

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(15213) # hardcode 



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        ttoc = time.time() - self.epoch_time_start
        self.times.append(ttoc)
        logging.info("time ------- " + str(ttoc))


def get_all_sr(path, ext):
    """
    store all *sr files in a list

    :params path: the directory path that contains *sr files
    """
    output_list = [os.path.join(path, element) for element in os.listdir(path) if element.endswith(ext)]
    logging.info ("Get " + str(len(output_list)) + " hdf5 objects...")
    # throw exception for the cases that no *sr is found in the path
    if len(output_list) == 0:
        logging.info ("warning: no *sr was found here at the generator..")
    return output_list


def get_sr_tuple(sr_list, random_split=1):
    """
    it takes a list of sr filenames, and return a tuple 
    in the format of (hdf5 object, index)

    :params sr_list: a list of sr filename
    :return: a list of tuple that stores (hdf5 object, index)
    """
    sr_tuple = []
    nslice_list = []
    len_sr = len(sr_list)
    #tic = time.time()
    for nid in range(len_sr):
        # load the number of slices from the hdf5 files
        hdf5_object = h5py.File(sr_list[nid], 'r')
        nslices = hdf5_object["nslice"][:][0]
        hdf5_object.close()
 
        random_nslice = random.sample(list(range(nslices)), nslices // random_split) # random version

        nslice_list.append(nslices) # non-random version # mainly used for finding the boundary, don't need to change

        # iterate all slices in the h5py
        #current_tuple = [(nid, nslice_id) for nslice_id in range(nslices)] # random version
        current_tuple = [(nid, nslice_id) for nslice_id in random_nslice]
        if sr_tuple == None:
            sr_tuple = current_tuple
        else:
            sr_tuple.extend(current_tuple)
    #print (time.time()-tic)
    return sr_list, sr_tuple, nslice_list


def get_25D_slice(path, index, nslice, max_slice, flag):
    """
    Takes a new hdf5 object and return a 2.5D slice;
    Note that for the edge cases, we just run repetitive. More precisely, 
    for the 1st slice, the 2.5D volumes with 5 slices is {1st,1st,1st,2nd,3rd}.

    :params hdf5: the hdf5 object
    :params index: the index of slices within the hdf5 files
    :params nslice: the number of slices in the 2.5D slice
    :params flag: whether it is in the prediciton mode; if flag == True, it is in the training mode; 
    i.e. it will generate two volumnes; otherwise, it will only generate a 2.5D volume for the testing
    """
    #print ("Get the 2.5D slice ...")
    tic = time.time()
    hdf5_object = h5py.File(path, 'r', libver='latest', swmr=True)
    #current_row, current_col = 512, 512
    # allocate the space for the 2.5D volume
    volume_in = []

    # get the central slices
    if flag:
        volume_out = np.expand_dims(hdf5_object["output"][index], axis=0)
        num_volume = len(volume_out.shape)
        if num_volume == 4:
            _, current_row, current_col, _ = volume_out.shape
            volume_in = np.zeros((nslice, current_row, current_col, 1), dtype=np.float32)
        elif num_volume == 5:
            _, current_row, current_col, current_slice, _ = volume_out.shape
            volume_in = np.zeros((nslice, current_row, current_col, current_slice, 1), dtype=np.float32)
        else : 
            pass
    else :
        hdf5_shape = len(hdf5_object["input"][index])
        if len(hdf5_shape) == 3:
            current_row, current_col, _ = hdf5_shape
            volume_in = np.zeros((nslice, current_row, current_col, 1), dtype=np.float32)
        elif len(hdf5_shape) == 4:
            current_row, current_col, current_slice, _ = hdf5_object["input"][index].shape
            volume_in = np.zeros((nslice, current_row, current_col, current_slice, 1), dtype=np.float32)
        else :
            pass

    # if the nslice == 1: use 2D
    if nslice == 1:
        volume_in = hdf5_object["input"][index]
        if flag : 
            return volume_in, volume_out
        else :
            return volume_in
    else:
        pass


class DataGenerator(keras.utils.Sequence):
    """
    Override the Data Generator
    """
    def __init__ (self, dirpath, img_size, ext, nslice = 5, batch_size=8, 
                                    shuffle=True, is_pred = False, verbose=True):
        # initialize the imgs
        #tic = time.time()
        logging.info ("--- Start run data generator ...")
        tic = time.time()

        self.img_size = img_size
        self.img_slice = 0
        self.t3d = False
        if len(self.img_size) == 4:
            self.t3d = True
            _, self.img_row, self.img_col, self.img_slice = img_size
        else :
            _, self.img_row, self.img_col = img_size

        self.nslice = nslice
        self.batch_size = batch_size

        # get all *sr files from the path
        sr_list = get_all_sr(dirpath, ext)

        # open all hdf5 files and generate a list of object
        self.hdf5_list, self.sr_tuple, self.slice_list = get_sr_tuple(sr_list)

        # calculate the length of the generator
        self.num_tuple = len(self.sr_tuple)
        self.shuffle_index = list(range(self.num_tuple))

        # run shuffle
        self.shuffle = shuffle

        # set mode
        self.is_pred = is_pred

        #logging.INFO("Shuffling the index again at the end of one epoch")
        self.on_epoch_end()
        #logging.INFO("Time cost at __init__ part" + str(time.time() - tic))
        logging.info ("--- Finish initialization ...: Time = " + str(time.time() - tic))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #ttoc = time.time() - self.epoch_time_start
        if self.shuffle == True:
            #logging.info("T:" +threading.get_ident() + "--- Run shuffling")
            self.shuffle_index = np.random.permutation(self.shuffle_index)
        #logging.info("generator time at each epoch ------- " + str(ttoc))


    def __len__ (self):
        return int(np.floor(self.num_tuple / self.batch_size))


    def __getitem__ (self, index):
        # extract the index range at current batch
        batch = self.shuffle_index[index * self.batch_size : (index+1) * self.batch_size]
        if self.t3d:
            batch_in = np.zeros((self.batch_size, self.img_row, self.img_col, self.img_slice, self.nslice), dtype=float)
            batch_out = np.zeros((self.batch_size, self.img_row, self.img_col, self.img_slice, 1), dtype=float)
        else:
            batch_in = np.zeros((self.batch_size, self.img_row, self.img_col, self.nslice), dtype=float)
            batch_out = np.zeros((self.batch_size, self.img_row, self.img_col, 1), dtype=float)

        if self.is_pred == False:
            # iterate batch
            for i in range(self.batch_size):
                hdf5_id, slice_id = self.sr_tuple[batch[i]]
                batch_in[i], batch_out[i] = get_25D_slice(self.hdf5_list[hdf5_id], slice_id, self.nslice, self.slice_list[hdf5_id], True)

            return batch_in, batch_out
        else :
            # iterate batch
            for i in range(self.batch_size):
                hdf5_id, slice_id = self.sr_tuple[batch[i]]
                batch_in[i] = get_25D_slice(self.hdf5_list[hdf5_id], slice_id, self.nslice, self.slice_list[hdf5_id], False)

            return batch_in

