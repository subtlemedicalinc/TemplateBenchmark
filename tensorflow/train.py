#!/usr/bin/env python3

"""
test pipeline 

@authors: Long Wang (long@subtlemedical.com)
Copyright (c) Subtle Medical, Inc.
"""

# sys packages
import argparse
import yaml
import os
import datetime
import time
import shutil
import sys
import logging
from time import strftime
from keras.utils import multi_gpu_model

# set up logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_name = strftime("log_%m%d%H%M.log")
logging.basicConfig(level=logging.INFO,
                    filename=log_name)

# keras-related package
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.optimizers import Adam

# utils packages
from utils.others import load_files
from keras import backend as K
from utils.generator import DataGenerator, TimeHistory

def create_parser():
    parser = argparse.ArgumentParser(description='Pipeline to run SR training ... ')
    parser.add_argument('--trc', type=str, help="Training configuration file [.yaml].")
    parser.add_argument('--mc', type=str, help="Model description file [.yaml].")
    parser.add_argument('--uid', type=str, help="sequence id for the experiment")
    parser.add_argument('--d', type=str, help="#id GPU device")
    parser.add_argument('--m', type=str, help='the training mode (t) or the inference mode (i)')
    parser.add_argument('--nslices', type=int, default=1, help='the number of slice we use to run 2.5D')
    parser.add_argument('--fit', type=str, help="generator: is to use fit_generator(); otherwise, use fit()")
    parser.add_argument('--gpus', type=int, default=1, help="multi-gpu")
    return parser


def run():
    # list all the argv
    logging.info("------------\n" + " ".join(sys.argv) + "\n-----------\n")

    # set args
    args = create_parser().parse_args()
    
    # setup the GPU
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.d
    logging.info("Applying GPU = " + args.d)

    # read model config file
    with open(args.mc, 'r') as f:
        model_config = yaml.load(f.read())
        
    # setup baseline
    from m0 import m0
    model = m0(**model_config)
    print (args.gpus)
    if args.gpus > 1:
        print ("applying multi-gpu model")
        model = multi_gpu_model(model, gpus=args.gpus)

    # read training file
    with open(args.trc, 'r') as f:
        train_config = yaml.load(f.read())

    if train_config["pretrain"] != "null":
       logging.info ("pre-trained weights is in " + train_config["pretrain"]) 
       model.load_weights(train_config["pretrain"], by_name = True)
    else :
       logging.info ("starting with the random initialization...")
  
    num_freeze_layers = int(train_config["freeze_layers"]) # should use try-catch
    if num_freeze_layers > 0:
       logging.info ("Start setting frozen layers")
       for layer in model.layers[:num_freeze_layers]:
           layer.trainable = False
       logging.debug (model.summary())

    # compile the model 
    model.compile(loss='mae', optimizer='adam')

    # select mode    
    if args.m == 'infer':
        run_inference(model, args.uid, zip_path, test_config) 
    elif args.m == 'train':
        # run training
        logging.info("starting to train...")
        img_col = model_config["img_col"]
        img_row = model_config["img_row"]
        img_channel = model_config["img_channel"]
        if model_config["img_slice"] > 0:
            img_slice = model_config["img_slice"]
            img_size = (img_channel, img_row, img_col, img_slice)
        else :
            img_size = (img_channel, img_row, img_col) 
        train(model=model, uuid=args.uid, train_config=train_config, 
            img_size=img_size, nslices=args.nslices, mode=args.fit)

    # clear the sessions
    K.clear_session()

def train(model, uuid, train_config, img_size, nslices, mode='generator'):
    cp_name = os.path.join(uuid  + "{epoch:02d}.h5") 
    csv_name = os.path.join(uuid + ".csv")

    # set up the metrics list
    csv_cb = CSVLogger(csv_name)
    # set up the checkpoint
    #ckpt_cb = ModelCheckpoint(cp_name, monitor='val_loss', save_best_only=False, save_weights_only=False, period = 1)

    time_callback = TimeHistory()
    callbacks = []
    #callbacks.append(ckpt_cb)
    callbacks.append(csv_cb)
    callbacks.append(time_callback)
    # running in the fit() mode
    if mode == "fit":
        tic = time.time()
        train_in, train_out = load_files(train_config["train_path"], train_config["ext"])
        valid_in, valid_out = load_files(train_config["valid_path"], train_config["ext"])
        logging.info("Time of the loading : " + str(time.time() - tic))

        logging.info ("start to train")
        tic = time.time()
        model.fit(train_in, train_out, 
                    batch_size=train_config["batch_size"],
                    epochs=train_config["epochs"],
                    verbose=train_config["verbose"],
                    shuffle=train_config['shuffle'],
                    validation_data = (valid_in, valid_out),
                    callbacks=callbacks)

        toc = time.time() - tic
        logging.info ("It takes " + str(toc) + " to train") 
        logging.info(time_callback.times)

    elif mode == "generator" :
        # running in the fit_generator mode
        logging.info("applying generator mode")

        tic = time.time()

        # generate a training generator
        training_generator = DataGenerator(dirpath=train_config["train_path"], 
                    img_size=img_size,
                    batch_size=train_config['batch_size'],
                    nslice=train_config["nslice"], ext=train_config["ext"])

        # generate a validation generator
        validation_generator = DataGenerator(dirpath=train_config["valid_path"],
                        img_size=img_size,
                        batch_size=train_config['batch_size'],
                        nslice=train_config["nslice"], 
                        ext=train_config["ext"])
        
        logging.info ("Time of the step to set up generator: " +  str(time.time() - tic))

        # set up the fit generator
        model.fit_generator(epochs=train_config["epochs"], 
                        generator=training_generator, 
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        max_queue_size=train_config["queue_size"], 
                        workers=train_config["workers"], 
                        use_multiprocessing=train_config["multi_process"], 
                        shuffle=train_config["shuffle"])

    return model

if __name__ == "__main__":
    run()



