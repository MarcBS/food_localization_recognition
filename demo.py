# -*- coding: utf-8 -*-

##########################################################################################
path_keras = '~/code/keras_marc-1.0.10'
path_keras_wrapper = '/media/HDD_2TB/marc/staged_keras_wrapper_v1'
##########################################################################################


import sys
for i, p in enumerate(sys.path):
    if 'keras' in p or 'staged_keras_wrapper' in p:
        del sys.path[i]
sys.path.append(path_keras)
sys.path.append(path_keras_wrapper)

from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.use('Agg') # run matplotlib without X server (GUI)
import matplotlib.pyplot as plt


# Old staged_keras_wrapper version
from Dataset import Dataset, saveDataset, loadDataset
from CNN_Model import CNN_Model, loadModel, saveModel
from ECOC_Classifier import ECOC_Classifier
from Stage import Stage
from Staged_Network import Staged_Network, saveStagedModel, loadStagedModel
from Utils import *

from localization_utilities import *

import os
import logging
import sys
import shutil
import ntpath
import fnmatch
import time
import numpy as np
from scipy import misc

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

'''
    Code dependencies

        - MarcBS/keras fork >= v1.0.10
                   https://github.com/MarcBS/keras
        - multimodal_keras_wrapper v0.05
                   https://github.com/MarcBS/multimodal_keras_wrapper/releases/tag/v0.05
        - anaconda
        - theano

    Usage

        Insert the paths of keras and keras_wrapper at the top of demo.py before running this demo.
        Edit load_parameters() for modifying the default parameters or input them as arguments
        with the following format:
            python demo.py param_name1=param_value1 param_name2=param_value2 ...

    Reference

        Bolaños, Marc, and Petia Radeva.
        "Simultaneous Food Localization and Recognition."
        arXiv preprint arXiv:1604.07953 (2016).
'''

############################################################
############################################################
#        MAIN FUNCTIONS
############################################################
############################################################
def load_parameters():

    ################# Input data parameters
    # path to a single image file that will be processed (only used if 'imgs_list' is empty)
    img = '/media/HDD_2TB/DATASETS/UECFOOD256/90/5512.jpg'

    # path to a .txt file containing a list of images to process
    imgs_list = ''

    dataset_ref_path = 'Datasets/Dataset_FoodVsNoFood.pkl'

    ################# Localization parameters
    # best params for combined datasets UECFood256 and EgocentricFood
    localization_params_t = 0.4 # minimum percentage allowed for considering a detection (aka 't' in reference paper)
    localization_params_s = 0.1 # remove all regions covering less than a certain percentage size of the original image (aka 's' in reference paper)
    localization_params_e = 0.2 # expand the bounding boxes by a certain percentage (aka 'e' in reference paper)
    n_top_convs = 0 # number of extra top-scoring convolutional maps used for localization (default=0, to use only main FAM)
    # path to the food-related objects localization model
    model_loc_path = 'Final_Models/recognition/GoogleNet_FoodVsNoFood_GAP'
    # If True then the it will try to load the CAMs from the results folder for generating new BBox proposals
    reuse_CAM = False

    ################# Recognition parameters
    apply_recognition = True
    # Food101
    model_rec_path = 'Final_Models/localization/GoogleNet_UECFood256/finetuning_Food101_NoFood'
    model_rec_iter = 27900
    # Egocentric Food
    #model_rec_path = '/media/HDD_2TB/CNN_MODELS/GoogleNet_EgocentricFood/finetuning_Food101_NoFood'
    #model_rec_iter = 9000

    ################# Other parameters
    # path to the folder where the results will be stored
    results_dir = '/media/HDD_2TB/marc/FoodDetection_keras/results/demo'

    batch_size = 25
    use_gpu = True

    # ==================================================
    default_args = locals().copy()
    return default_args


def preprocessParams(params):
    params['img'] = str(params['img'])
    params['imgs_list'] = str(params['imgs_list'])
    params['dataset_ref_path'] = str(params['dataset_ref_path'])
    params['localization_params_t'] = float(params['localization_params_t'])
    params['localization_params_s'] = float(params['localization_params_s'])
    params['localization_params_e'] = float(params['localization_params_e'])
    params['n_top_convs'] = int(params['n_top_convs'])
    params['model_loc_path'] = str(params['model_loc_path'])
    params['reuse_CAM'] = bool(params['reuse_CAM'])
    params['apply_recognition'] = bool(params['apply_recognition'])
    params['model_rec_path'] = str(params['model_rec_path'])
    params['model_rec_iter'] = int(params['model_rec_iter'])
    params['results_dir'] = str(params['results_dir'])
    params['batch_size'] = int(params['batch_size'])
    params['use_gpu'] = bool(params['use_gpu'])

    return params

def main(params, localization_net=None, W=None, recognition_net=None):

    params = preprocessParams(params)
    logging.info('Using the following params:')
    print params

    if os.path.exists(params['results_dir']) and not params['reuse_CAM']:
        shutil.rmtree(params['results_dir'])
    if not params['reuse_CAM']:
        os.makedirs(params['results_dir'])

    ############################################################
    # Load default dataset for applying images pre-processing
    ds = loadDataset(params['dataset_ref_path'])
    # Get list of images
    if params['imgs_list'] == '':
        logging.info('Applying localization on a single image.')
        imgs_list = [params['img']]
    else:
        logging.info('Applying localization on list of images.')
        imgs_list = listPaths(params['imgs_list'])
    ############################################################

    ############################################################
    # Load localization model (GAP model)
    if localization_net is None:
        logging.info('Loading food localization model...')
        localization_net = loadStagedModel(params['model_loc_path'])
    # Prepare network for generating CAMs and get matrix for weighting
    if W is None:
        W = prepareCAM(localization_net)
    # Load recognition model
    if params['apply_recognition'] and recognition_net is None:
        logging.info('Loading food recognition model...')
        recognition_net = loadModel(params['model_rec_path'], params['model_rec_iter'])
    ############################################################

    # Prepare structure for storing results
    results_struct = dict()
    results_struct['bboxes'] = []
    results_struct['scores'] = []
    results_struct['labels'] = []
    results_struct['imgs_paths'] = []

    ############################################################
    start_time = time.time()
    eta = -1
    # Start images processing
    n_imgs = len(imgs_list)
    predictions_binary = np.zeros((n_imgs))
    for init in range(0, n_imgs, params['batch_size']):

        # Load batch of test images and their pre-processed version (X)
        final = min(init + params['batch_size'], n_imgs)
        this_list = imgs_list[init:final]
        [X, original_sizes] = loadBatchImages(ds, this_list)

        # Compute CAMs (class activation maps)
        reshape_size = [256, 256]
        if not params['reuse_CAM']:
            [maps, pred, convs] = computeCAM(localization_net, X, W, reshape_size=reshape_size, n_top_convs=params['n_top_convs'])
            predictions_binary[init:final] = pred
        else:
            logging.info("Reusing previously calculated CAMs")
            maps = np.zeros((X.shape[0], W.shape[1], reshape_size[0], reshape_size[1]))
            convs = np.zeros((X.shape[0], W.shape[1], params['n_top_convs'], reshape_size[0], reshape_size[1]))
            for i in range(final - init):
                im_name = '%0.6d' % (init + i)
                map_name = im_name + '_CAM.npy'
                #maps[i][1] = misc.imread(params['results_dir'] + '/' + map_name)
                maps[i][1] = np.load(params['results_dir'] + '/' + map_name)
                for i_conv in range(params['n_top_convs']):
                    conv_name = im_name + '_Conv_' + str(i_conv) + '.npy'
                    #convs[i, 1, i_conv] = misc.imsave(params['results_dir'] + '/' + conv_name)
                    convs[i, 1, i_conv] = np.load(params['results_dir'] + '/' + conv_name)

        # Process each image separately
        for i in range(maps.shape[0]):
            predicted_bboxes = []
            predicted_scores = []
            predicted_Y = []

            if pred[i]: # only continue if the current image has been predicted as Food
                in_CAMS = [maps[i][1]]
                for i_conv in range(params['n_top_convs']):
                    in_CAMS.append(convs[i, 1, i_conv])

                # Apply food localization
                [predicted_bboxes, predicted_scores] = getBBoxesFromCAMs(in_CAMS, reshape_size=original_sizes[i],
                                                                        percentage_heat=params['localization_params_t'],
                                                                        size_restriction=params['localization_params_s'],
                                                                        box_expansion=params['localization_params_e'],
                                                                        use_gpu=params['use_gpu'])

                # Apply food recognition
                if params['apply_recognition']:
                    [predicted_bboxes, predicted_scores, predicted_Y] = recognizeBBoxes(this_list[i], predicted_bboxes,
                                                                                    recognition_net, ds,
                                                                                    remove_non_food=None)

            results_struct['bboxes'].append(predicted_bboxes)
            results_struct['scores'].append(predicted_scores)
            results_struct['labels'].append(predicted_Y)
            results_struct['imgs_paths'].append(this_list[i])


        # Save maps
        if not params['reuse_CAM']:
            for i in range(final - init):
                im_name = '%0.6d' % (init + i)
                map_name = im_name + '_CAM.npy'
                # index 1 identifies the FAM (food-activation map)
                #misc.imsave(params['results_dir'] + '/' + map_name, maps[i][1]) 
                np.save(params['results_dir'] + '/' + map_name, maps[i][1])

                for i_conv in range(params['n_top_convs']):
                    conv_name = im_name + '_Conv_' + str(i_conv) + '.npy'
                    #misc.imsave(params['results_dir'] + '/' + conv_name, convs[i, 1, i_conv])
                    np.save(params['results_dir'] + '/' + conv_name, convs[i, 1, i_conv])

        eta = (n_imgs - final) * (time.time() - start_time) / final
        sys.stdout.write('\r')
        sys.stdout.write("Processed %d/%d images -  ETA: %ds " % (final, n_imgs, int(eta)))
        sys.stdout.flush()
    print
    ############################################################

    ############################################################
    # Save additional data
    if not params['reuse_CAM']:
        np.save(params['results_dir'] + '/' + 'predictions_binary.npy', predictions_binary)
    np.save(params['results_dir'] + '/' + 'results_loc_rec.npy', results_struct)
    np.save(params['results_dir'] + '/' + 'parameters.npy', params)
    logging.info('Done')
    logging.info('Results stored in '+params['results_dir'])
    ############################################################


############################################################
############################################################
#        AUXILIARY FUNCTIONS
############################################################
############################################################

def listPaths(file):
    imgs_list = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            imgs_list.append(line)
    return imgs_list


def loadBatchImages(ds, list):
    original_sizes = []
    new_list = []

    # Load set of images
    nImages = len(list)
    for i in range(nImages):
        im = list[i]

        # Check if the filename includes the extension
        [path, filename] = ntpath.split(im)
        [filename, ext] = os.path.splitext(filename)

        # If it doesn't then we find it
        if (not ext):
            filename = fnmatch.filter(os.listdir(path), filename + '*')
            if (not filename):
                raise Exception('Non existent image ' + im)
            else:
                im = path + '/' + filename[0]

        # Read image
        im = misc.imread(im)
        original_sizes.append(im.shape[:2])
        new_list.append(im)

    X = ds.loadImages(new_list, normalization=False, meanSubstraction=True, dataAugmentation=False,
                      external=True, loaded=True)
    return [X, original_sizes]


if __name__ == '__main__':

    default_args = load_parameters()

    args = {}
    for arg in sys.argv[1:]:
        try:
            k, v = arg.split('=')
        except:
            print 'Arguments must have the following format: param_name=param_val'
            exit(1)
        if k in default_args.keys():
            default_args[k] = v
        else:
            raise Exception('Non-valid argument provided with key "'+k+'"')

    sys.exit(main(default_args))
