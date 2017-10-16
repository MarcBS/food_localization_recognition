# -*- coding: utf-8 -*-

from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.use('Agg') # run matplotlib without X server (GUI)
import matplotlib.pyplot as plt

'''
from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
from keras_wrapper.cnn_model import CNN_Model, loadModel, saveModel
from keras_wrapper.ecoc_classifier import ECOC_Classifier
from keras_wrapper.stage import Stage
from keras_wrapper.staged_network import Staged_Network, saveStagedModel, loadStagedModel
from keras_wrapper.utils import *
'''
# Old wrapper version
from Dataset import Dataset, saveDataset, loadDataset
from CNN_Model import CNN_Model, loadModel, saveModel
from ECOC_Classifier import ECOC_Classifier
from Stage import Stage
from Staged_Network import Staged_Network, saveStagedModel, loadStagedModel
from Utils import *

from localization_utilities import *
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

from keras.models import Sequential, Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU

# Import Selective Search code
selsearch_root = '/home/lifelogging/code/selective_search'
import sys
sys.path.insert(0, selsearch_root)
import sel_search

import os
import logging
import copy
from operator import add
import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
from skimage.transform import resize

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def main():
    """
        Best validation accuracy of 0.9564 was achieved on iteration 24.000
    """
    
    #startTraining()
    #resumeTraining()
    #testBestNet()
    
    #startTrainingGAP()
    #testGAPModel()
    #testGAPLocalization()
    
    
    ''' Process UECFood-256 dataset '''
    
    #generateCAMsUECTest()
    #generateWindowsSelSearchTest()
    #generateWindowsFasterRCNNTest()
    #crossValidationPrecRec()
    
    
    ''' Process EgocentricFood dataset '''
    #generateCAMsEgoFoodTest()
    #generateWindowsSelSearchEgoFoodTest()
    #generateWindowsFasterRCNNEgoFoodTest()
    #crossValidationEgoFoodPrecRec()
    #crossValidationEgoFoodPrecRecInitFin(351, 550)
    #crossValidationEgoFoodPrecRecInitFin(551, 750)

    
    ''' Process both datasets simultaneously '''
    #evalCombined()
    #evalCombinedPrecisionRecall_fixedIoU()



################################################
#
#    Food vs NoFood model functions
#
################################################


def startTraining():
    
    logging.info('#### Start Food Vs NoFood network training. ####')
        
    training_parameters = {'n_epochs': 20, 'batch_size': 50, 'report_iter': 50, 'iter_for_val': 3000, 
                            'lr_decay': 6000, 'lr_gamma': 0.75, 'save_model': 3000}
                            
    learning_rate = 0.001
    
    # Load dataset
    ds = loadFoodVsNoFoodDataset()
    ds.setTrainMean(mean_image = [122.6795, 116.6690, 104.0067])
    saveDataset(ds, 'Datasets')
    
    
    # Load GoogLeNet
    net = loadGoogleNet()
    prepareGoogleNet_Food101(net)
    
    # Modify softmax output
    net.removeLayers(['loss3/classifier','output_loss3/loss3'])
    net.removeOutputs(['loss3/loss3'])
    net.model.add_node(Dense(2, activation='softmax'), name='loss3/classifier_food_vs_nofood', input='loss3/classifier_flatten')
    net.model.add_output(name='loss3/loss3', input='loss3/classifier_food_vs_nofood')
    
    # Compile
    net.setOptimizer(lr=learning_rate)
    
    # Insert into staged network
    snet = Staged_Network(model_name='GoogleNet_FoodVsNoFood', plots_path='Plots/GoogleNet_FoodVsNoFood', models_path='Models/GoogleNet_FoodVsNoFood')
    snet.addStage(net, out_name='loss3/loss3', balanced=True)
    
    # Train net
    snet.trainNet(ds, 0, training_parameters)
    
    # Save model
    saveStagedModel(snet)
    
    # Test net
    test_params = {'batch_size': 100};
    snet.testNet(ds)

    
    
def resumeTraining():
    
    logging.info('#### Resume Food Vs NoFood network training. ####')
        
    training_parameters = {'n_epochs': 20, 'batch_size': 50, 'report_iter': 50, 'iter_for_val': 3000, 
                            'lr_decay': 6000, 'lr_gamma': 0.1, 'save_model': 3000}
                            
    learning_rate = 0.001
    
    # Load dataset
    ds = loadFoodVsNoFoodDataset()


    # Recover solver state
    net = loadModel('Models/GoogleNet_FoodVsNoFood/Stage_0/Branch_0', 6000)
    
    # Compile
    net.setOptimizer(lr=learning_rate)
    
    # Insert into staged network
    snet = Staged_Network(model_name='GoogleNet_FoodVsNoFood_resumed', plots_path='Plots/GoogleNet_FoodVsNoFood_resumed', models_path='Models/GoogleNet_FoodVsNoFood_resumed')
    snet.addStage(net, out_name='loss3/loss3', balanced=True)
    
    # Resume training net
    snet.resumeTrainNet(ds, 0, training_parameters)
    
    # Save model
    saveStagedModel(snet)
    
    # Test net
    test_params = {'batch_size': 100};
    snet.testNet(ds)
    
    
def testBestNet():
    ntests = 10
    batch_size = 100
    
    logging.info('#### Test best Food Vs NoFood network. ####')
    
    # Load dataset
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    
    # Recover best model
    net = loadModel('Models/GoogleNet_FoodVsNoFood_resumed/Stage_0/Branch_0', 24000)
    
    logging.info('Best validation accuracy: 0.9564')
    
    # Test net
    scores = []
    for i in range(ntests):
        [X, Y] = ds.getXY('val', batch_size, normalization=False, meanSubstraction=True, dataAugmentation=False)
        
        (loss, score, top_score) = net.testOnBatch(X, Y, out_name='loss3/loss3')
        scores.append(float(score))
    logging.info('Validation accuracy on first %s validation samples was %s' % (str(ntests*batch_size), str(np.mean(scores))))



################################################
#
#    GAP model functions
#
################################################

def evalCombined():
    ##################################################################
    #     Parameters

    # Recognition model
    #model_path_rec = ['/media/HDD_2TB/CNN_MODELS/GoogleNet_UECFood256/finetuning_Food101_NoFood', '/media/HDD_2TB/CNN_MODELS/GoogleNet_EgocentricFood/finetuning_Food101_NoFood']
    model_path_rec = None
    iter_rec = [27900, 9000]
    n_classes = [257, 10]

    # Detection test images
    path_imgs = ['/media/HDD_2TB/DATASETS/UECFOOD256', '/media/HDD_2TB/DATASETS/EgocentricFood']
    #test_list = 'val_list.txt'
    test_list = ['test_list.txt', 'test_list.txt']

    #detect_type = 'GAP'
    #detect_type = 'SelSearch'
    detect_type = 'FasterRCNN'

    # Result
    #maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test_ego']
    #maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_ego']
    maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_ego']

    #results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test_combined'
    #results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_combined'
    results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_combined'

    ##### Cross-validation parameters

    # FoodVsNoFood GoogleNet GAP parameter search
#    params = dict()
#    params['percentage_heat'] = 0.4
#    params['size_restriction'] = 0.1
#    params['box_expansion'] = 0.2
#    params['n_bboxes'] = 999999
#    iou_values = np.arange(0.5,1.0001,0.1)

    # Selective Search or Faster-RCNN parameter search
    params = dict()
    params['percentage_heat'] = 0
    params['size_restriction'] = 0
    params['box_expansion'] = 0
    params['n_bboxes'] = 999999
    iou_values = np.arange(0.5,1.0001,0.1)

    ##################################################################

    predictions = []
    ds = []
    samples_detection = []
    net = []
    for i in range(len(maps_dir)): # load data and predictions for each dataset
        # Load dataset for pre-processing test images
        ds_ = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
        # Load UEC detection test set
        [ds_, sd] = loadUECDetectionTest(ds_, path_imgs[i], test_list[i])
        ds.append(ds_)
        samples_detection.append(sd)

        if(detect_type == 'GAP'):
            # Load predictions from GAP model
            predictions.append(np.load(maps_dir[i]+'/predictions.npy'))
        else:
            predictions.append([])

        # Load recognition model
        if(model_path_rec is None):
            net.append(None) # not applying recognition right now
        else:
            net.append(loadModel(model_path_rec[i], iter_rec[i]))
    
    
    # Arrays for storing results and parameters for each test
    prec_list = []
    rec_list = []
    acc_list = []
    num_GT = []
    num_predictions = []
    
    prec_list_classes = []
    rec_list_classes = []
    acc_list_classes = []
    num_GT_classes = []
    num_predictions_classes = []
    
    params_list = []
    reports = []
    
    report = None
    for iou in iou_values:
    
        logging.info('Applying combined test...')
                        
        # Prepare parameters
        params['IoU'] = iou
        
        # Compute evaluation measures
        [general_measures, class_measures, report] = computePrecisionRecall(net, n_classes, ds, maps_dir, \
                                                                            samples_detection, predictions, \
                                                                            params, detect_type=detect_type, \
                                                                            report_all=report, use_gpu=True, combined=True)
        
        [prec, rec, acc, total_GT, total_pred] = general_measures
        [prec_classes, rec_classes, acc_classes, total_GT_classes, total_pred_classes] = class_measures
        
        # Store results and parameters
        prec_list.append(prec)
        rec_list.append(rec)
        acc_list.append(acc)
        num_GT.append(total_GT)
        num_predictions.append(total_pred)
        
        prec_list_classes.append(prec_classes)
        rec_list_classes.append(rec_classes)
        acc_list_classes.append(acc_classes)
        num_GT_classes.append(total_GT_classes)
        num_predictions_classes.append(total_pred_classes)
        
        params_list.append([params['percentage_heat'], params['size_restriction'], params['box_expansion'], params['IoU'], params['n_bboxes']])
        reports.append(report) # [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] for each image
                        
                    
    # Store results
    logging.info('Storing results')
    if(not os.path.isdir(results_dir)):
        os.makedirs(results_dir)
    np.save(results_dir+'/precisions_'+detect_type+'.npy', prec_list)
    np.save(results_dir+'/recalls_'+detect_type+'.npy', rec_list)
    np.save(results_dir+'/accuracies_'+detect_type+'.npy', acc_list)
    np.save(results_dir+'/num_GT_'+detect_type+'.npy', num_GT)
    np.save(results_dir+'/num_predictions_'+detect_type+'.npy', num_predictions)
    
    np.save(results_dir+'/precisions_classes_'+detect_type+'.npy', prec_list_classes)
    np.save(results_dir+'/recalls_classes_'+detect_type+'.npy', rec_list_classes)
    np.save(results_dir+'/accuracies_classes_'+detect_type+'.npy', acc_list_classes)
    np.save(results_dir+'/num_GT_classes_'+detect_type+'.npy', num_GT_classes)
    np.save(results_dir+'/num_predictions_classes_'+detect_type+'.npy', num_predictions_classes)
    
    np.save(results_dir+'/params_cross_val_'+detect_type+'.npy', params_list)
    np.save(results_dir+'/reports_'+detect_type+'.npy', reports)
    
    logging.info('Cross-validation finished')

    
def evalCombinedPrecisionRecall_fixedIoU():
    ##################################################################
    #     Parameters

    # Recognition model
    n_classes = [257, 10]

    # Detection test images
    path_imgs = ['/media/HDD_2TB/DATASETS/UECFOOD256', '/media/HDD_2TB/DATASETS/EgocentricFood']
    #test_list = 'val_list.txt'
    test_list = ['test_list.txt', 'test_list.txt']

    
    # min_prediction_score thresholds for calculating the Precision-Recall curve for a fixedIoU value
    thresholds = np.arange(0,1.01,0.01) # np.arange(0,1,0.1) # (too fast: 10s)
    fixedIoU=0.5
    
    
    #detect_type = 'GAP'
    #detect_type = 'SelSearch'
    detect_type = 'FasterRCNN'

    # Result
    #maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test_ego']
    #maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_ego']
    maps_dir = ['/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test', '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_ego']

    #results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test_combined'
    #results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_combined'
    results_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_combined'

    ##### Cross-validation parameters (NOT USED IN THIS FUNCTION)

    # FoodVsNoFood GoogleNet GAP parameter search
#    params = dict()
#    params['percentage_heat'] = 0.4
#    params['size_restriction'] = 0.1
#    params['box_expansion'] = 0.2
#    params['n_bboxes'] = 999999
#    iou_values = np.arange(0.5,1.0001,0.1)

    # Selective Search or Faster-RCNN parameter search
#    params = dict()
#    params['percentage_heat'] = 0
#    params['size_restriction'] = 0
#    params['box_expansion'] = 0
#    params['n_bboxes'] = 999999
#    iou_values = np.arange(0.5,1.0001,0.1)

    ##################################################################

    samples_detection = []
    for i in range(len(maps_dir)): # load data and predictions for each dataset
        # Load dataset for pre-processing test images
        ds_ = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
        # Load UEC detection test set
        [ds_, sd] = loadUECDetectionTest(ds_, path_imgs[i], test_list[i])
        samples_detection.append(sd)
    
    
    # Arrays for storing results and parameters for each test
    prec_list = []
    rec_list = []
    prec_list_classes = []
    rec_list_classes = []
    

    logging.info('Applying combined test fixedIoU...')
                        
    report = np.load(results_dir+'/reports_'+detect_type+'.npy')
    params = np.load(results_dir+'/params_cross_val_'+detect_type+'.npy')
    # find position of the chosen fixedIoU
    pos_IoU = None
    for pos_p in range(len(params)):
        if params[pos_p][3] == fixedIoU:
            pos_IoU = pos_p
    if pos_IoU is None:
        print "params", params
        print "fixedIoU", fixedIoU
        raise Exception("pos_IoU not found!")
    report = report[pos_IoU]
        
    # Compute evaluation measures
    [general_measures, class_measures] = computePrecisionRecall_fixedIoU(n_classes, samples_detection, report,
                                                                         fixedIoU=fixedIoU, thresholds=thresholds,
                                                                         combined=True)
    
    # Get results for each min_prediction_score value
    n_thresholds = len(thresholds)
    for thres in range(n_thresholds):
        [prec, rec, acc, total_GT, total_pred] = general_measures[thres]
        [prec_classes, rec_classes, acc_classes, total_GT_classes, total_pred_classes] = class_measures[thres]
        
        # Store results and parameters
        prec_list.append(prec)
        rec_list.append(rec)
        prec_list_classes.append(prec_classes)
        rec_list_classes.append(rec_classes)
                        
                    
    # Store results
    logging.info('Storing results')

    np.save(results_dir+'/fixedIoU_'+str(fixedIoU)+'_precisions_'+detect_type+'.npy', prec_list)
    np.save(results_dir+'/fixedIoU_'+str(fixedIoU)+'_recalls_'+detect_type+'.npy', rec_list)
    np.save(results_dir+'/fixedIoU_'+str(fixedIoU)+'_precisions_classes_'+detect_type+'.npy', prec_list_classes)
    np.save(results_dir+'/fixedIoU_'+str(fixedIoU)+'_recalls_classes_'+detect_type+'.npy', rec_list_classes)
    
    logging.info('Precision-Recall calculation on fixedIoU value finished')
    
    

def startTrainingGAP():
    
    logging.info('#### Start Food Vs NoFood GAP (localization) network training. ####')
        
        
    new_model_name = 'GoogleNet_FoodVsNoFood_GAP'
    training_parameters = {'n_epochs': 1, 'batch_size': 50, 'report_iter': 50, 'iter_for_val': 500, 
                            'lr_decay': 500, 'lr_gamma': 0.1, 'save_model': 500}
                            
    learning_rate = 0.000001
    
    # Load dataset
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    
    # Recover best model
    net = loadModel('Models/GoogleNet_FoodVsNoFood_resumed/Stage_0/Branch_0', 24000)
    
    # Modify network
    snet = generateGAPforGoogleNet(net, new_model_name, learning_rate)
    
    # Save staged network
    saveStagedModel(snet)
    
    # Train net
    snet.trainNet(ds, 1, training_parameters)
    
    # Save model
    saveStagedModel(snet)
    
    # Test net
    test_params = {'batch_size': 100};
    snet.testNet(ds, parameters=test_params)
    
    logging.info('Done')



def testGAPModel():
    new_model_name = 'GoogleNet_FoodVsNoFood_GAP'
    
    # Load dataset
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    
    # Load model
    snet = loadStagedModel('Models/'+new_model_name)

    # Test net
    test_params = {'batch_size': 100};
    snet.testNet(ds, parameters=test_params)
    
    logging.info('Done')
    
    
def testGAPLocalization():
    
    ############################
    #     Parameters
    
    model_path = '/media/HDD_2TB/CNN_MODELS/GoogleNet_FoodVsNoFood_GAP'
    reshape_size = [256,256]
    
    # List of test images
    path_imgs = 'images'
    list_imgs = ['1-eating.jpg', '2-eating.jpg', '3-eating.jpg', '4-eating.jpg', '5-eating.jpg', '6-eating.jpg', '7-eating.jpg',
        '8-eating.jpg', '9-eating.jpg', '10-eating.jpg', '1-noeating.jpg', '2-noeating.jpg', '3-noeating.jpg', '4-noeating.jpg', '5-noeating.jpg']
     
    ############################
    raise NotImplementedError()
    
    # NMS and multiple convolutional features need to be included
#    # Now apply NMS on all the obtained bboxes
#    dets = np.hstack((np.array(predicted_bboxes), np.array(predicted_scores)[:, np.newaxis])).astype(np.float32)
#    if(use_gpu):
#        keep = gpu_nms(dets, 0.3, device_id=0)
#    else:
#        keep = cpu_nms(dets, 0.3)
#    dets = dets[keep, :]
#    predicted_bboxes = []
#    predicted_scores = []
#    for idet in range(dets.shape[0]):
#        predicted_bboxes.append(dets[idet, :4])
#        predicted_scores.append(dets[idet, -1])
        
        
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')

    # Load GAP model
    snet = loadStagedModel(model_path)
    
    # Prepare network for generating CAMs and get matrix for weighting
    W = prepareCAM(snet)

    # Load test images and their pre-processed version (X)
    list_ = [path_imgs+'/'+im for im in list_imgs]
    [images, X] = loadImagesExternal(ds, list_)
   
    # Compute CAMs (class activation maps)
    #[maps, predictions] = computeCAM(snet, X, W, reshape_size)
    [maps, predictions, convs] = computeCAM(snet, X, W, reshape_size)
        
    # Store resulting images
    cmap = plt.get_cmap('jet') # set colormap
    for s in range(X.shape[0]):
        plt.figure(1)
        count = 1
        orig = resize(images[s], tuple(reshape_size), order=1, preserve_range=True)
        ax = plt.subplot(1, W.shape[1]+1, count)
        count += 1
        ax.set_title(str(predictions[s]))
        plt.imshow(np.array(orig, dtype=np.uint8))
        for c in range(W.shape[1]): # plot each class
            norm = (maps[s][c] - np.min(maps[s][c])) / (np.max(maps[s][c]) - np.min(maps[s][c])) # 0-1 normalization
            heat = cmap(norm) # apply colormap
            heat = np.delete(heat, 3, 2) # ???
            heat = np.array(heat[:, :, :3]*255, dtype=np.uint8)
            img = np.array(orig*0.3 + heat*0.7, dtype=np.uint8)
            ax = plt.subplot(1, W.shape[1]+1, count)
            count += 1
            ax.set_title(str(c))
            plt.imshow(img)
        # Save figure
        plot_file = list_[s]+'_Heatmap.jpg'
        plt.savefig(plot_file)
        # Close plot window
        plt.close()

    logging.info('Done!')


def generateCAMsUECTest():
    '''
        Generates and stores the CAMs from the UECFood-256 detection set.
    '''

    ############################
    #     Parameters
    
    # Detection model
    model_path = '/media/HDD_2TB/CNN_MODELS/GoogleNet_FoodVsNoFood_GAP'
    batch_size = 25
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/UECFOOD256'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_val'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test'
    
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_val'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_test'
    save_map = 1 # position of the map to save (1 = Food)
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    

    # Load GAP model
    snet = loadStagedModel(model_path)
    
    # Prepare network for generating CAMs and get matrix for weighting
    W = prepareCAM(snet)

    # Load and process images in batches
    logging.info('Start computing CAM for each image.')
    n_imgs = ds.len_test
    predictions = np.zeros((n_imgs))
    for init in range(0, n_imgs, batch_size):
        
        # Load batch of test images and their pre-processed version (X)
        final = min(init+batch_size, n_imgs)
        [images, X] = loadImagesDataset(ds, init, final)
       
        # Compute CAMs (class activation maps)
        #[maps, pred] = computeCAM(snet, X, W)
        [maps, pred, convs] = computeCAM(snet, X, W)
        predictions[init:final] = pred
        
        # Recover best convolutional features per image and class
        #convs = getBestConvFeatures(snet, X, W, n_top_convs=20)
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_CAM.jpg'
            misc.imsave(maps_dir+'/'+map_name, maps[i][save_map])
            
            for i_conv in range(convs.shape[2]):
                conv_name = im_name+'_Conv_'+str(i_conv)+'.jpg'
                misc.imsave(maps_dir+'/'+conv_name, convs[i,save_map,i_conv])
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' CAMs.')
    
    # Save additional data
    np.save(maps_dir+'/'+'predictions.npy', predictions)
    logging.info('Done')



def generateCAMsEgoFoodTest():
    '''
        Generates and stores the CAMs from the EgocentricFood detection set.
    '''

    ############################
    #     Parameters
    
    # Detection model
    model_path = '/media/HDD_2TB/CNN_MODELS/GoogleNet_FoodVsNoFood_GAP'
    batch_size = 25
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/EgocentricFood'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_val_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_test_ego'
    
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_val_ego'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_test_ego'
    save_map = 1 # position of the map to save (1 = Food)
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    

    # Load GAP model
    snet = loadStagedModel(model_path)
    
    # Prepare network for generating CAMs and get matrix for weighting
    W = prepareCAM(snet)

    # Load and process images in batches
    logging.info('Start computing CAM for each image.')
    n_imgs = ds.len_test
    predictions = np.zeros((n_imgs))
    for init in range(0, n_imgs, batch_size):
        
        # Load batch of test images and their pre-processed version (X)
        final = min(init+batch_size, n_imgs)
        [images, X] = loadImagesDataset(ds, init, final)
       
        # Compute CAMs (class activation maps)
        #[maps, pred] = computeCAM(snet, X, W)
        [maps, pred, convs] = computeCAM(snet, X, W)
        predictions[init:final] = pred
        
        # Recover best convolutional features per image and class
        #convs = getBestConvFeatures(snet, X, W, n_top_convs=20)
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_CAM.jpg'
            misc.imsave(maps_dir+'/'+map_name, maps[i][save_map])
            
            for i_conv in range(convs.shape[2]):
                conv_name = im_name+'_Conv_'+str(i_conv)+'.jpg'
                misc.imsave(maps_dir+'/'+conv_name, convs[i,save_map,i_conv])
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' CAMs.')
    
    # Save additional data
    np.save(maps_dir+'/'+'predictions.npy', predictions)
    logging.info('Done')



def generateWindowsSelSearchTest():
    '''
        Generates and stores the Selective Search windows from the UECFood-256 detection set.
    '''

    ############################
    #     Parameters
    batch_size = 25
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/UECFOOD256'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_val'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test'
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    

    # Load and process images in batches
    logging.info('Start computing SelSearch for each image.')
    n_imgs = ds.len_test
    for init in range(0, n_imgs, batch_size):
        
        # Get batch of images
        final = min(init+batch_size, n_imgs)
        images = ds.X_test[init:final]

        # Compute SelSearch windows
        windows_ = sel_search.get_windows(images)
        windows = []
        for im in windows_: # change y position by x position
            windows.append([[w[1], w[0], w[3], w[2]] for w in im])
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_windows.npy'
            np.save(maps_dir+'/'+map_name, windows[i])
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' SelSearch windows.')
    
    logging.info('Done')
      

def generateWindowsSelSearchEgoFoodTest():
    '''
        Generates and stores the Selective Search windows from the EgocentricFood detection set.
    '''

    ############################
    #     Parameters
    batch_size = 25
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/EgocentricFood'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_val_ego'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_ego'
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    

    # Load and process images in batches
    logging.info('Start computing SelSearch for each image.')
    n_imgs = ds.len_test
    for init in range(0, n_imgs, batch_size):
        
        # Get batch of images
        final = min(init+batch_size, n_imgs)
        images = ds.X_test[init:final]

        # Compute SelSearch windows
        windows_ = sel_search.get_windows(images)
        windows = []
        for im in windows_: # change y position by x position
            windows.append([[w[1], w[0], w[3], w[2]] for w in im])
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_windows.npy'
            np.save(maps_dir+'/'+map_name, windows[i])
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' SelSearch windows.')
    
    logging.info('Done')



def generateWindowsFasterRCNNTest():
    '''
        Generates and stores the Faster-RCNN windows from the UECFood-256 detection set.
    '''
    import _init_paths
    from fast_rcnn.config import cfg
    from fast_rcnn.test import im_detect
    from fast_rcnn.nms_wrapper import nms
    import caffe

    ############################
    #     Parameters
    #batch_size = 25
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/UECFOOD256'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Network parameters
    prototxt = '/home/lifelogging/code/faster_rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/lifelogging/code/faster_rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
    gpu_id = 0
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_val'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test'
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    
    
    # Load net 
    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    

    # Load and process images in batches
    logging.info('Start computing Faster-RCNN for each image.')
    n_imgs = ds.len_test
    batch_size = 1
    for init in range(0, n_imgs, batch_size):
        
        # Get batch of images
        final = min(init+batch_size, n_imgs)
        #[images, X] = loadImagesDataset(ds, init, final)
        
        # batch_size forced to 1
        images = ds.X_test[init:final]
        images = misc.imread(images[0])
        # convert RBG to BGR
        images = images[:, :, (2, 1, 0)]

        # Compute Faster-RCNN windows
        scores, boxes = im_detect(net, images)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        windows = dets
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_windows.npy'
            np.save(maps_dir+'/'+map_name, windows)
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' Faster-RCNN windows.')
    
    logging.info('Done')



def generateWindowsFasterRCNNEgoFoodTest():
    '''
        Generates and stores the Faster-RCNN windows from the EgocentricFood detection set.
    '''
    import _init_paths
    from fast_rcnn.config import cfg
    from fast_rcnn.test import im_detect
    from fast_rcnn.nms_wrapper import nms
    import caffe

    ############################
    #     Parameters
    #batch_size = 25
    #batch_size = 1
    
    # Detection images
    #set_name = 'val'
    set_name = 'test'
    path_imgs = '/media/HDD_2TB/DATASETS/EgocentricFood'
    #test_list = 'val_list.txt'
    test_list = 'test_list.txt'
    
    # Network parameters
    prototxt = '/home/lifelogging/code/faster_rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    caffemodel = '/home/lifelogging/code/faster_rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
    gpu_id = 0
    
    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_val_ego'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_ego'
    ############################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    
    
    # Load net 
    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    

    # Load and process images in batches
    logging.info('Start computing Faster-RCNN for each image.')
    n_imgs = ds.len_test
    batch_size = 1
    for init in range(0, n_imgs, batch_size):
        
        # Get batch of images
        final = min(init+batch_size, n_imgs)
        #[images, X] = loadImagesDataset(ds, init, final)
                
        # batch_size forced to 1
        images = ds.X_test[init:final]
        images = misc.imread(images[0])
        # convert RBG to BGR
        images = images[:, :, (2, 1, 0)]

        # Compute Faster-RCNN windows
        scores, boxes = im_detect(net, images)
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        windows = dets
        
        # Save maps
        for i in range(final-init):
            im_name = samples_detection['all_ids'][init+i]
            map_name = im_name+'_windows.npy'
            np.save(maps_dir+'/'+map_name, windows)
            
        logging.info('Computed '+str(final) +'/'+ str(n_imgs) + ' Faster-RCNN windows.')
    
    logging.info('Done')




################################################
#
#    GAP Evaluation functions
#
################################################


def crossValidationPrecRec():
    '''
        Applies a cross validation for generating a precision-recall curve
    '''
    
    ##################################################################
    #     Parameters
    
    # Recognition model
    model_path_rec = '/media/HDD_2TB/CNN_MODELS/GoogleNet_UECFood256/finetuning_Food101_NoFood'
    iter_rec = 27900
    n_classes = 257

    # Detection test images
    path_imgs = '/media/HDD_2TB/DATASETS/UECFOOD256'
    test_list = 'val_list.txt'
    #test_list = 'test_list.txt'
    
    detect_type = 'GAP'
    #detect_type = 'SelSearch'
    #detect_type = 'FasterRCNN'

    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_val'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_val'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_val'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test'

    use_gpu = True
    
    # Cross-validation parameters
    parameters = dict()
    
    # FoodVsNoFood GoogleNet GAP parameter search
    parameters['percentage_heat'] = np.arange(0.2, 1.01, 0.2)
    parameters['size_restriction'] = np.arange(0.02, 0.1001, 0.02)
    parameters['box_expansion'] = np.arange(0.2, 1.01, 0.2)
    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,21)

    # GAP second round
#    parameters['percentage_heat'] = np.arange(0.2, 1.01, 0.2)
#    parameters['size_restriction'] = [0.0]
#    parameters['box_expansion'] = np.arange(0.2, 1.01, 0.2)
#    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
#    parameters['n_bboxes'] = [9999]


    # Selective Search or Faster-RCNN parameter search
#    parameters['percentage_heat'] = [0]
#    parameters['size_restriction'] = [0]
#    parameters['box_expansion'] = [0]
#    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
#    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,1000,10)

    ##################################################################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    if(detect_type == 'GAP'):
        # Load predictions from GAP model
        predictions = np.load(maps_dir+'/predictions.npy')
    else:
        predictions = []

    # Load recognition model
    net = None # not applying recognition right now
    #net = loadModel(model_path_rec, iter_rec)
    
    # Arrays for storing results and parameters for each test
    prec_list = []
    rec_list = []
    acc_list = []
    num_GT = []
    num_predictions = []
    
    prec_list_classes = []
    rec_list_classes = []
    acc_list_classes = []
    num_GT_classes = []
    num_predictions_classes = []
    
    params_list = []
    reports = []
    
    # Apply cross-validation test on all the samples in the validation set
    n_rounds = len(parameters['percentage_heat']) * len(parameters['size_restriction']) * len(parameters['box_expansion']) * \
               len(parameters['iou_values']) * len(parameters['n_bboxes'])
    round = 1
    for per_heat in parameters['percentage_heat']:
        for size_res in parameters['size_restriction']:
            for box_exp in parameters['box_expansion']:
                for n_bb in parameters['n_bboxes']:
                    report = None
                    for iou in parameters['iou_values']:
                    
                        logging.info('Applying CV round '+str(round) + '/' + str(n_rounds))
                        
                        # Prepare parameters
                        params = dict()
                        params['percentage_heat'] = per_heat
                        params['size_restriction'] = size_res
                        params['box_expansion'] = box_exp
                        params['IoU'] = iou
                        params['n_bboxes'] = n_bb
                        
                        # Compute evaluation measures
                        [general_measures, class_measures, report] = computePrecisionRecall(net, n_classes, ds, maps_dir, \
                                                                                            samples_detection, predictions, \
                                                                                            params, detect_type=detect_type, \
                                                                                            report_all=report, use_gpu=use_gpu)
                        
                        [prec, rec, acc, total_GT, total_pred] = general_measures
                        [prec_classes, rec_classes, acc_classes, total_GT_classes, total_pred_classes] = class_measures
                        
                        # Store results and parameters
                        prec_list.append(prec)
                        rec_list.append(rec)
                        acc_list.append(acc)
                        num_GT.append(total_GT)
                        num_predictions.append(total_pred)
                        
                        prec_list_classes.append(prec_classes)
                        rec_list_classes.append(rec_classes)
                        acc_list_classes.append(acc_classes)
                        num_GT_classes.append(total_GT_classes)
                        num_predictions_classes.append(total_pred_classes)
                        
                        params_list.append([per_heat, size_res, box_exp, iou, n_bb])
                        reports.append(report) # [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] for each image
                        
                        
                        if(round % 50 == 0):
                            logging.info('Storing results backup')
                            np.save(maps_dir+'/precisions_backup_'+str(round)+'_'+detect_type+'.npy', prec_list)
                            np.save(maps_dir+'/recalls_backup_'+str(round)+'_'+detect_type+'.npy', rec_list)
                            np.save(maps_dir+'/accuracies_backup_'+str(round)+'_'+detect_type+'.npy', acc_list)
                            np.save(maps_dir+'/num_GT_backup_'+str(round)+'_'+detect_type+'.npy', num_GT)
                            np.save(maps_dir+'/num_predictions_backup_'+str(round)+'_'+detect_type+'.npy', num_predictions)
                            
                            np.save(maps_dir+'/precisions_classes_backup_'+str(round)+'_'+detect_type+'.npy', prec_list_classes)
                            np.save(maps_dir+'/recalls_classes_backup_'+str(round)+'_'+detect_type+'.npy', rec_list_classes)
                            np.save(maps_dir+'/accuracies_classes_backup_'+str(round)+'_'+detect_type+'.npy', acc_list_classes)
                            np.save(maps_dir+'/num_GT_classes_backup_'+str(round)+'_'+detect_type+'.npy', num_GT_classes)
                            np.save(maps_dir+'/num_predictions_classes_backup_'+str(round)+'_'+detect_type+'.npy', num_predictions_classes)
                            
                            np.save(maps_dir+'/params_cross_val_backup_'+str(round)+'_'+detect_type+'.npy', params_list)
                            np.save(maps_dir+'/reports_backup_'+str(round)+'_'+detect_type+'.npy', reports)
                        
                        round += 1
                    
    # Store results
    logging.info('Storing results')
    np.save(maps_dir+'/precisions_'+detect_type+'.npy', prec_list)
    np.save(maps_dir+'/recalls_'+detect_type+'.npy', rec_list)
    np.save(maps_dir+'/accuracies_'+detect_type+'.npy', acc_list)
    np.save(maps_dir+'/num_GT_'+detect_type+'.npy', num_GT)
    np.save(maps_dir+'/num_predictions_'+detect_type+'.npy', num_predictions)
    
    np.save(maps_dir+'/precisions_classes_'+detect_type+'.npy', prec_list_classes)
    np.save(maps_dir+'/recalls_classes_'+detect_type+'.npy', rec_list_classes)
    np.save(maps_dir+'/accuracies_classes_'+detect_type+'.npy', acc_list_classes)
    np.save(maps_dir+'/num_GT_classes_'+detect_type+'.npy', num_GT_classes)
    np.save(maps_dir+'/num_predictions_classes_'+detect_type+'.npy', num_predictions_classes)
    
    np.save(maps_dir+'/params_cross_val_'+detect_type+'.npy', params_list)
    np.save(maps_dir+'/reports_'+detect_type+'.npy', reports)
    
    logging.info('Cross-validation finished')




def crossValidationEgoFoodPrecRec():
    '''
        Applies a cross validation for generating a precision-recall curve
    '''
    
    ##################################################################
    #     Parameters
    
    # Recognition model
    #model_path_rec = '/media/HDD_2TB/CNN_MODELS/GoogleNet_UECFood256/finetuning_Food101_NoFood'
    #iter_rec = 27900
    #n_classes = 257
    n_classes = 10

    # Detection test images
    path_imgs = '/media/HDD_2TB/DATASETS/EgocentricFood'
    test_list = 'val_list.txt'
    #test_list = 'test_list.txt'
    
    detect_type = 'GAP'
    #detect_type = 'SelSearch'
    #detect_type = 'FasterRCNN'

    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_val_ego'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_val_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_val_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_ego'

    use_gpu = True

    # Cross-validation parameters
    parameters = dict()
    
    # FoodVsNoFood GoogleNet GAP parameter search
    parameters['percentage_heat'] = np.arange(0.2, 1.01, 0.2)
    parameters['size_restriction'] = np.arange(0.02, 0.1001, 0.02)
    parameters['box_expansion'] = np.arange(0.2, 1.01, 0.2)
    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,21)


    # Selective Search or Faster-RCNN parameter search
#    parameters['percentage_heat'] = [0]
#    parameters['size_restriction'] = [0]
#    parameters['box_expansion'] = [0]
#    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
#    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,1000,10)

    ##################################################################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    if(detect_type == 'GAP'):
        # Load predictions from GAP model
        predictions = np.load(maps_dir+'/predictions.npy')
    else:
        predictions = []

    # Load recognition model
    net = None # not applying recognition right now
    #net = loadModel(model_path_rec, iter_rec)
    
    # Arrays for storing results and parameters for each test
    prec_list = []
    rec_list = []
    acc_list = []
    num_GT = []
    num_predictions = []
    
    prec_list_classes = []
    rec_list_classes = []
    acc_list_classes = []
    num_GT_classes = []
    num_predictions_classes = []
    
    params_list = []
    reports = []
    
    # Apply cross-validation test on all the samples in the validation set
    n_rounds = len(parameters['percentage_heat']) * len(parameters['size_restriction']) * len(parameters['box_expansion']) * \
               len(parameters['iou_values']) * len(parameters['n_bboxes'])
    round = 1
    for per_heat in parameters['percentage_heat']:
        for size_res in parameters['size_restriction']:
            for box_exp in parameters['box_expansion']:
                for n_bb in parameters['n_bboxes']:
                    report = None
                    for iou in parameters['iou_values']:
                    
                        logging.info('Applying CV round '+str(round) + '/' + str(n_rounds))
                        
                        # Prepare parameters
                        params = dict()
                        params['percentage_heat'] = per_heat
                        params['size_restriction'] = size_res
                        params['box_expansion'] = box_exp
                        params['IoU'] = iou
                        params['n_bboxes'] = n_bb
                        
                        # Compute evaluation measures
                        [general_measures, class_measures, report] = computePrecisionRecall(net, n_classes, ds, maps_dir, \
                                                                                            samples_detection, predictions, \
                                                                                            params, detect_type=detect_type, \
                                                                                            report_all=report, use_gpu=use_gpu)
                        
                        [prec, rec, acc, total_GT, total_pred] = general_measures
                        [prec_classes, rec_classes, acc_classes, total_GT_classes, total_pred_classes] = class_measures
                        
                        # Store results and parameters
                        prec_list.append(prec)
                        rec_list.append(rec)
                        acc_list.append(acc)
                        num_GT.append(total_GT)
                        num_predictions.append(total_pred)
                        
                        prec_list_classes.append(prec_classes)
                        rec_list_classes.append(rec_classes)
                        acc_list_classes.append(acc_classes)
                        num_GT_classes.append(total_GT_classes)
                        num_predictions_classes.append(total_pred_classes)
                        
                        params_list.append([per_heat, size_res, box_exp, iou, n_bb])
                        reports.append(report) # [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] for each image
                        
                        
                        if(round % 50 == 0):
                            logging.info('Storing results backup')
                            np.save(maps_dir+'/precisions_backup_'+str(round)+'_'+detect_type+'.npy', prec_list)
                            np.save(maps_dir+'/recalls_backup_'+str(round)+'_'+detect_type+'.npy', rec_list)
                            np.save(maps_dir+'/accuracies_backup_'+str(round)+'_'+detect_type+'.npy', acc_list)
                            np.save(maps_dir+'/num_GT_backup_'+str(round)+'_'+detect_type+'.npy', num_GT)
                            np.save(maps_dir+'/num_predictions_backup_'+str(round)+'_'+detect_type+'.npy', num_predictions)
                            
                            np.save(maps_dir+'/precisions_classes_backup_'+str(round)+'_'+detect_type+'.npy', prec_list_classes)
                            np.save(maps_dir+'/recalls_classes_backup_'+str(round)+'_'+detect_type+'.npy', rec_list_classes)
                            np.save(maps_dir+'/accuracies_classes_backup_'+str(round)+'_'+detect_type+'.npy', acc_list_classes)
                            np.save(maps_dir+'/num_GT_classes_backup_'+str(round)+'_'+detect_type+'.npy', num_GT_classes)
                            np.save(maps_dir+'/num_predictions_classes_backup_'+str(round)+'_'+detect_type+'.npy', num_predictions_classes)
                            
                            np.save(maps_dir+'/params_cross_val_backup_'+str(round)+'_'+detect_type+'.npy', params_list)
                            np.save(maps_dir+'/reports_backup_'+str(round)+'_'+detect_type+'.npy', reports)
                        
                        round += 1
                    
    # Store results
    logging.info('Storing results')
    np.save(maps_dir+'/precisions_'+detect_type+'.npy', prec_list)
    np.save(maps_dir+'/recalls_'+detect_type+'.npy', rec_list)
    np.save(maps_dir+'/accuracies_'+detect_type+'.npy', acc_list)
    np.save(maps_dir+'/num_GT_'+detect_type+'.npy', num_GT)
    np.save(maps_dir+'/num_predictions_'+detect_type+'.npy', num_predictions)
    
    np.save(maps_dir+'/precisions_classes_'+detect_type+'.npy', prec_list_classes)
    np.save(maps_dir+'/recalls_classes_'+detect_type+'.npy', rec_list_classes)
    np.save(maps_dir+'/accuracies_classes_'+detect_type+'.npy', acc_list_classes)
    np.save(maps_dir+'/num_GT_classes_'+detect_type+'.npy', num_GT_classes)
    np.save(maps_dir+'/num_predictions_classes_'+detect_type+'.npy', num_predictions_classes)
    
    np.save(maps_dir+'/params_cross_val_'+detect_type+'.npy', params_list)
    np.save(maps_dir+'/reports_'+detect_type+'.npy', reports)
    
    logging.info('Cross-validation finished')




def crossValidationEgoFoodPrecRecInitFin(init_tests, finish_tests):
    '''
        Applies a cross validation for generating a precision-recall curve
    '''
    
    ##################################################################
    #     Parameters
    
    # Recognition model
    #model_path_rec = '/media/HDD_2TB/CNN_MODELS/GoogleNet_UECFood256/finetuning_Food101_NoFood'
    #iter_rec = 27900
    #n_classes = 257
    n_classes = 10

    # Detection test images
    path_imgs = '/media/HDD_2TB/DATASETS/EgocentricFood'
    test_list = 'val_list.txt'
    #test_list = 'test_list.txt'
    
    detect_type = 'GAP'
    #detect_type = 'SelSearch'
    #detect_type = 'FasterRCNN'

    # Result
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_val_ego'
    maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/CAMs_v2_val_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_val_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/SelSearch_test_ego'
    #maps_dir = '/media/HDD_2TB/marc/FoodDetection_keras_Data/FasterRCNN_test_ego'

    use_gpu = True

    # Cross-validation parameters
    parameters = dict()
    
    # FoodVsNoFood GoogleNet GAP parameter search
    parameters['percentage_heat'] = np.arange(0.2, 1.01, 0.2)
    parameters['size_restriction'] = np.arange(0.02, 0.1001, 0.02)
    parameters['box_expansion'] = np.arange(0.2, 1.01, 0.2)
    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,21)


    # Selective Search or Faster-RCNN parameter search
#    parameters['percentage_heat'] = [0]
#    parameters['size_restriction'] = [0]
#    parameters['box_expansion'] = [0]
#    parameters['iou_values'] = np.arange(0.5,1.0001,0.1)
#    parameters['n_bboxes'] = [9999]
##    parameters['n_bboxes'] = np.arange(0,1000,10)

    ##################################################################
    
    # Load dataset for pre-processing test images
    ds = loadDataset('Datasets/Dataset_FoodVsNoFood.pkl')
    # Load UEC detection test set
    [ds, samples_detection] = loadUECDetectionTest(ds, path_imgs, test_list)
    if(detect_type == 'GAP'):
        # Load predictions from GAP model
        predictions = np.load(maps_dir+'/predictions.npy')
    else:
        predictions = []

    # Load recognition model
    net = None # not applying recognition right now
    #net = loadModel(model_path_rec, iter_rec)
    
    # Arrays for storing results and parameters for each test
    prec_list = []
    rec_list = []
    acc_list = []
    num_GT = []
    num_predictions = []
    
    prec_list_classes = []
    rec_list_classes = []
    acc_list_classes = []
    num_GT_classes = []
    num_predictions_classes = []
    
    params_list = []
    reports = []
    
    # Apply cross-validation test on all the samples in the validation set
    n_rounds = len(parameters['percentage_heat']) * len(parameters['size_restriction']) * len(parameters['box_expansion']) * \
               len(parameters['iou_values']) * len(parameters['n_bboxes'])
    round = 1
    for per_heat in parameters['percentage_heat']:
        for size_res in parameters['size_restriction']:
            for box_exp in parameters['box_expansion']:
                for n_bb in parameters['n_bboxes']:
                    report = None
                    for iou in parameters['iou_values']:
                    
                        if(round >= init_tests and round <= finish_tests):
                    
                            logging.info('Applying CV round '+str(round) + '/' + str(n_rounds))
                            
                            # Prepare parameters
                            params = dict()
                            params['percentage_heat'] = per_heat
                            params['size_restriction'] = size_res
                            params['box_expansion'] = box_exp
                            params['IoU'] = iou
                            params['n_bboxes'] = n_bb
                            
                            # Compute evaluation measures
                            [general_measures, class_measures, report] = computePrecisionRecall(net, n_classes, ds, maps_dir, \
                                                                                                samples_detection, predictions, \
                                                                                                params, detect_type=detect_type, \
                                                                                                report_all=report, use_gpu=use_gpu)
                            
                            [prec, rec, acc, total_GT, total_pred] = general_measures
                            [prec_classes, rec_classes, acc_classes, total_GT_classes, total_pred_classes] = class_measures
                            
                            # Store results and parameters
                            prec_list.append(prec)
                            rec_list.append(rec)
                            acc_list.append(acc)
                            num_GT.append(total_GT)
                            num_predictions.append(total_pred)
                            
                            prec_list_classes.append(prec_classes)
                            rec_list_classes.append(rec_classes)
                            acc_list_classes.append(acc_classes)
                            num_GT_classes.append(total_GT_classes)
                            num_predictions_classes.append(total_pred_classes)
                            
                            params_list.append([per_heat, size_res, box_exp, iou, n_bb])
                            reports.append(report) # [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] for each image
                            
                            
                            if(round % 50 == 0):
                                logging.info('Storing results backup')
                                np.save(maps_dir+'/precisions_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', prec_list)
                                np.save(maps_dir+'/recalls_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', rec_list)
                                np.save(maps_dir+'/accuracies_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', acc_list)
                                np.save(maps_dir+'/num_GT_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', num_GT)
                                np.save(maps_dir+'/num_predictions_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', num_predictions)
                                
                                np.save(maps_dir+'/precisions_classes_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', prec_list_classes)
                                np.save(maps_dir+'/recalls_classes_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', rec_list_classes)
                                np.save(maps_dir+'/accuracies_classes_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', acc_list_classes)
                                np.save(maps_dir+'/num_GT_classes_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', num_GT_classes)
                                np.save(maps_dir+'/num_predictions_classes_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', num_predictions_classes)
                                
                                np.save(maps_dir+'/params_cross_val_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', params_list)
                                np.save(maps_dir+'/reports_backup_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+str(round)+'_'+detect_type+'.npy', reports)
                            
                        round += 1
                    
    # Store results
    logging.info('Storing results')
    np.save(maps_dir+'/precisions_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', prec_list)
    np.save(maps_dir+'/recalls_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', rec_list)
    np.save(maps_dir+'/accuracies_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', acc_list)
    np.save(maps_dir+'/num_GT_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', num_GT)
    np.save(maps_dir+'/num_predictions_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', num_predictions)
    
    np.save(maps_dir+'/precisions_classes_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', prec_list_classes)
    np.save(maps_dir+'/recalls_classes_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', rec_list_classes)
    np.save(maps_dir+'/accuracies_classes_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', acc_list_classes)
    np.save(maps_dir+'/num_GT_classes_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', num_GT_classes)
    np.save(maps_dir+'/num_predictions_classes_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', num_predictions_classes)
    
    np.save(maps_dir+'/params_cross_val_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', params_list)
    np.save(maps_dir+'/reports_init_'+str(init_tests)+'_fin_'+str(finish_tests)+'_'+detect_type+'.npy', reports)
    
    logging.info('Cross-validation finished')


def computePrecisionRecall(net, n_classes, ds, maps_dir, samples_detection, predictions, params, report_all=None, \
                           detect_type='GAP', use_gpu=True, combined=False):
    '''
        Computes a Precision-Recall curve and its associated mAP score given a "net" model for recognition, 
        a test dataset stored in "ds", its associated information in "samples_detection" and
        a set of parameters in the dictionary structure "params". "predictions" say if the net found
        any food in the test image or not.
        The set of parameters provided in "params" must be:
            percentage_heat : [0,1] threshold of minimum percentage of the max heat (to be considered food)
            size_restriction : [0,1] minimum percentage restriction to consider a bbox before its expansion
            box_expansion : [0,1] percentage of box expansion before final prediction
            IoU : [0,1] minimum valid intersection over union value for considering a positive localization
            n_bboxes : int, max number of bounding boxes per CAM (used for computing the Average Precision)
        "samples_detection" is a dictionary that should contain:
            all_ids : all unique image ids
            list_imgs : all unique images
            original_sizes : sizes from original images
            match_ids : indices to matching samples
            classes : classes from each bounding box
            boxes : bounding boxes
        The parameter "report_all" can be provided for skipping the recomputation of the bounding boxes (the report
        corresponds to the last variable in the output of this function.
    '''
    #n_bboxes = 9999 # max number of bounding boxes per CAM

    if(not combined):
        net = [net]
        n_classes = [n_classes]
        ds = [ds]
        maps_dir = [maps_dir]
        samples_detection = [samples_detection]
        predictions = [predictions]

    n_bboxes = params['n_bboxes']
    percentage_heat = params['percentage_heat']
    size_restriction = params['size_restriction']
    box_expansion = params['box_expansion']
    IoU = params['IoU']

    n_samples = []
    for s in samples_detection:
        n_samples.append(len(s['all_ids']))

    # Counters for computing general precision-recall curve
    FP = 0
    TP = 0
    FN = 0
    total_GT = 0
    total_pred = 0
    
    # Counters for computing class-specific precision-recall curve
    n_tot = sum(n_classes) - (len(n_classes)-1)
    TP_classes = np.zeros((n_tot))
    FP_classes = np.zeros((n_tot))
    FN_classes = np.zeros((n_tot))
    total_GT_classes = np.zeros((n_tot))
    total_pred_classes = np.zeros((n_tot))
    
    
    if(report_all is None):
        report_all = []
        for d in range(len(n_samples)):
            report_all.append([])
        compute_report = True
    else:
        compute_report = False
    
    # Process each image's CAM
    n_datasets = len(n_samples)
    for dataset in range(n_datasets):
        for s in range(n_samples[dataset]):
            
            if(compute_report):
            
                # Get list of GT boxes and classes for the current image
                GT_bboxes = []
                GT_Y = []
                for m_, m in enumerate(samples_detection[dataset]['match_ids']):
                    if(m == s):
                        GT_bboxes.append(samples_detection[dataset]['boxes'][m_])
                        GT_Y.append(samples_detection[dataset]['classes'][m_])
            
                apply_recognition = False
                if(detect_type == 'GAP'): # Use FoodVsNoFood GoogleNet-GAP model for recognition
            
                    predicted_bboxes = []
                    predicted_Y = []
                    predicted_scores = []
                    
                    # Get all computed maps (if we are also using convolutional features)
                    all_maps = [maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_CAM.jpg']
                    i_conv = 0
                    next_map = maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_Conv_'+ str(i_conv) +'.jpg'
                    while(os.path.isfile(next_map) and i_conv < 10):
                        all_maps.append(next_map)
                        i_conv += 1
                        next_map = maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_Conv_'+ str(i_conv) +'.jpg'
                    
                    for map_path in all_maps:
            
                        #map = misc.imread(maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_CAM.jpg') # CAM only
                        map = misc.imread(map_path) # CAM and convolutional features
                        new_reshape_size = samples_detection[dataset]['original_sizes'][s][:2]

                        # Resize map to original size
                        map = resize(map, tuple(new_reshape_size), order=1, preserve_range=True)

                        # Detect regions above a certain percentage of the max heat
                        bb_thres = np.max(map)*percentage_heat

                        # Compute binary selected region
                        binary_heat = map
                        binary_heat = np.where(binary_heat>bb_thres, 255, 0)

                        # Get biggest connected component
                        min_size = new_reshape_size[0] * new_reshape_size[1] * size_restriction
                        labeled, nr_objects = ndimage.label(binary_heat) # get connected components
                        [objects, counts] = np.unique(labeled, return_counts=True) # count occurrences
                        biggest_components = np.argsort(counts[1:])[::-1]
                        selected_components = [1 if counts[i+1] >= min_size else 0 for i in biggest_components] # check minimum size restriction
                        biggest_components = biggest_components[:min([np.sum(selected_components), 9999])] # get all bboxes

                        # Extract each component (which will become a bbox prediction)
                        map = map/255.0 # normalize map
                        
                        # Only continue if the current image has been predicted as Food!
                        if(predictions[dataset][s] == 1):
                            for selected, comp in zip(selected_components, biggest_components):
                                if(selected):
                                    max_heat = np.where(labeled == comp+1, 255, 0) # get the biggest

                                    # Draw bounding box on original image
                                    box = list(bbox(max_heat))

                                    # expand box before final detection
                                    x_exp = box[2] * box_expansion
                                    y_exp = box[3] * box_expansion
                                    box[0] = max([0, box[0]-x_exp/2])
                                    box[1] = max([0, box[1]-y_exp/2])
                                    # change width and height by xmax and ymax
                                    box[2] += box[0]
                                    box[3] += box[1]
                                    box[2] = min([new_reshape_size[1]-1, box[2] + x_exp])
                                    box[3] = min([new_reshape_size[0]-1, box[3] + y_exp])

                                    # Apply recognition on the obtained box??
                                    if(net[dataset] != None): # we have a network loaded and have to compute predictions
                                        # Predict class
                                        apply_recognition = True
                                    else:
                                        # Do not use class predictions
                                        score = np.mean(map[box[1]:box[3],box[0]:box[2]]) # use mean CAM value of the bbox as a score
                                        predicted_scores.append(score)
                                    predicted_bboxes.append(box)
                                    
                    # Now apply NMS on all the obtained bboxes
                    nms_threshold = 0.3
                    #logging.info('bboxes before NMS: '+str(len(predicted_scores)))
                    if(len(predicted_scores) > 0):
                        dets = np.hstack((np.array(predicted_bboxes), np.array(predicted_scores)[:, np.newaxis])).astype(np.float32)
                        if(use_gpu):
                            keep = gpu_nms(dets, nms_threshold, device_id=0)
                        else:
                            keep = cpu_nms(dets, nms_threshold)
                        dets = dets[keep, :]
                        predicted_bboxes = []
                        predicted_scores = []
                        for idet in range(dets.shape[0]):
                            predicted_bboxes.append(dets[idet, :4])
                            predicted_scores.append(dets[idet, -1])
                    #logging.info('bboxes after NMS: '+str(len(predicted_scores)))
                                
                elif(detect_type == 'SelSearch'): # Use Selective Search model for recognition
                    #windows = sel_search.get_windows([samples_detection[dataset]['list_imgs'][s]])
                    #predicted_bboxes = windows[0]
                    predicted_Y = []
                    predicted_bboxes = np.load(maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_windows.npy')
                    predicted_scores = [1 for ss_box in range(len(predicted_bboxes))]
                elif(detect_type == 'FasterRCNN'): # Use Faster-RCNN model for recognition
                    predicted_Y = []
                    faster_bboxes = np.load(maps_dir[dataset]+'/'+samples_detection[dataset]['all_ids'][s]+'_windows.npy')
                    predicted_bboxes = []
                    predicted_scores = []
                    for i_fast in range(faster_bboxes.shape[0]):
                        predicted_bboxes.append(faster_bboxes[i_fast, :4])
                        predicted_scores.append(faster_bboxes[i_fast, -1])
                        #predicted_scores.append(1)
                else:
                    raise NotImplementedError()
                
                # Apply prediction on bounding boxes
                if(apply_recognition and len(predicted_bboxes) > 0):
                    # Load crops
                    im = misc.imread(samples_detection[dataset]['list_imgs'][s])
                    images_list = []
                    for b in predicted_bboxes:
                        images_list.append(im[b[1]:b[3], b[0]:b[2]])
                    
                    # Forward pass
                    X = ds[dataset].loadImages(images_list, normalization=False, meanSubstraction=True, dataAugmentation=False, loaded=True)
                    predictions_rec = net[dataset].predictOnBatch(X)['loss3/loss3']
                    
                    # Store prediction
                    max_pred = np.argmax(predictions_rec, axis=1)
                    final_bboxes = []
                    for im in range(predictions_rec.shape[0]):
                        # Remove bounding box prediction if we consider it is "NoFood"
                        if(max_pred[im] != 0):
                            predicted_Y.append(max_pred[im])
                            predicted_scores.append(predictions_rec[im][max_pred[im]])
                            final_bboxes.append(predicted_bboxes[im])
                    predicted_bboxes = final_bboxes
                
                # Report result for each image
                report_all[dataset].append([predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y])
                
            # Re-use report of the current image provided in the parameters (or recently computed)
            [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] = report_all[dataset][s]
            
            
            # Filter top scoring n_bboxes bounding boxes
            n_pred = min(len(predicted_scores), n_bboxes)
            predicted_bboxes = predicted_bboxes[:n_pred]
            predicted_Y = predicted_Y[:n_pred]
            predicted_scores = predicted_scores[:n_pred]
            
            
            # Compute TPs, FPs and FNs
            [TP_, FP_, FN_, TP_c, FP_c, FN_c] = computeMeasures(IoU, n_classes[dataset], predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y)
            total_GT += len(GT_bboxes)
            total_pred += len(predicted_bboxes)
            TP += TP_
            FP += FP_
            FN += FN_
            
            n_prev_classes = sum(n_classes[:dataset]) - (len(n_classes[:dataset]) - 1) - 1
            for pos in range(len(TP_c)):
                if(pos == 0):
                    TP_classes[pos] += TP_c[pos]
                    FP_classes[pos] += FP_c[pos]
                    FN_classes[pos] += FN_c[pos]
                else:
                    TP_classes[pos+n_prev_classes-1] += TP_c[pos]
                    FP_classes[pos+n_prev_classes-1] += FP_c[pos]
                    FN_classes[pos+n_prev_classes-1] += FN_c[pos]
                    
            for y in GT_Y:
                if(y == 0):
                    total_GT_classes[y] += 1
                else:
                    total_GT_classes[y+n_prev_classes] += 1
            for y in predicted_Y:
                if(y == 0):
                    total_pred_classes[y] += 1
                else:
                    total_pred_classes[y+n_prev_classes] += 1
                
        
    
    # Compute general precision - recall
    if((TP+FP) == 0):
        precision = 0.0
    else:
        precision = float(TP)/(TP+FP)
    if((TP+FN) == 0):
        recall = 0.0
    else:
        recall = float(TP)/(TP+FN)
    if((FP+FN+TP) == 0):
        accuracy = 0.0
    else:
        accuracy = float(TP)/(FP+FN+TP)
        
        
    # Compute class-specific precision - recall
    precision_classes = np.zeros((n_tot))
    recall_classes = np.zeros((n_tot))
    accuracy_classes = np.zeros((n_tot))
    for c in range(n_tot):
        TP = TP_classes[c]
        FP = FP_classes[c]
        FN = FN_classes[c]
        if((TP+FP) == 0):
            precision_classes[c] = 0.0
        else:
            precision_classes[c] = float(TP)/(TP+FP)
        if((TP+FN) == 0):
            recall_classes[c] = 0.0
        else:
            recall_classes[c] = float(TP)/(TP+FN)
        if((FP+FN+TP) == 0):
            accuracy_classes[c] = 0.0
        else:
            accuracy_classes[c] = float(TP)/(FP+FN+TP)
        
    general_measures = [precision, recall, accuracy, total_GT, total_pred]
    class_measures = [precision_classes, recall_classes, accuracy_classes, total_GT_classes, total_pred_classes]
    
    return [general_measures, class_measures, report_all]
  


def computePrecisionRecall_fixedIoU(n_classes, samples_detection, report_all, fixedIoU=0.5, 
                                    thresholds=np.arange(0,1,0.1), combined=False):
    '''
        Computes a Precision-Recall curve and its associated mAP score given a set of precalculated reports.
        The parameter "report_all" must include the following information for each sample: 
            [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y]
        The parameter 'threshods' defines the range of min_prediction_scores to be tested for computing the 
        precision-recall curve.
    '''
    #n_bboxes = 9999 # max number of bounding boxes per CAM

    if(not combined):
        n_classes = [n_classes]
        samples_detection = [samples_detection]

    n_samples = []
    for s in samples_detection:
        n_samples.append(len(s['all_ids']))
    
    n_thresholds = len(thresholds)
    n_datasets = len(n_samples)
    
    # prepare variables for storing all precision-recall values
    general_measures = [[] for j in range(n_thresholds)]
    class_measures = [[] for j in range(n_thresholds)]
    
    
    # compute precision-recall measures for each min_prediction_score threshold
    for thres in range(n_thresholds):
        
        # Counters for computing general precision-recall curve
        FP = 0
        TP = 0
        FN = 0
        total_GT = 0
        total_pred = 0

        # Counters for computing class-specific precision-recall curve
        n_tot = sum(n_classes) - (len(n_classes)-1)
        TP_classes = np.zeros((n_tot))
        FP_classes = np.zeros((n_tot))
        FN_classes = np.zeros((n_tot))
        total_GT_classes = np.zeros((n_tot))
        total_pred_classes = np.zeros((n_tot))
        
        # Process each image's CAM
        for dataset in range(n_datasets):
        
            for s in range(n_samples[dataset]):

                # Re-use report of the current image provided in the parameters (or recently computed)
                [predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y] = report_all[dataset][s]


                # Filter bounding boxes which are above the current threshold 'thres'
                aux_predicted_bboxes = []
                aux_predicted_Y = []
                aux_predicted_scores = []
                #if thresholds[thres] == 1.0:
                #    print "predicted_bboxes",predicted_bboxes
                #    print "predicted_Y",predicted_Y
                #    print "predicted_scores",predicted_scores
                for pos, score in enumerate(predicted_scores):
                    #if thresholds[thres] == 1.0:
                    #    print "pos",pos
                    #    print "score",score
                    #    print "thres",thresholds[thres]
                    if score > thresholds[thres]:
                        aux_predicted_bboxes.append(predicted_bboxes[pos])
                        aux_predicted_Y.append(predicted_Y[pos])
                        aux_predicted_scores.append(predicted_scores[pos])
                predicted_bboxes = aux_predicted_bboxes
                predicted_Y = aux_predicted_Y
                predicted_scores = aux_predicted_scores
                #if thresholds[thres] == 1.0:
                #    print "predicted_bboxes",predicted_bboxes
                #    print "predicted_Y",predicted_Y
                #    print "predicted_scores",predicted_scores

                # Compute TPs, FPs and FNs
                [TP_, FP_, FN_, TP_c, FP_c, FN_c] = computeMeasures(fixedIoU, n_classes[dataset], predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y)
                
                if thresholds[thres] == 1.0:
                    print "TP", TP_
                    print "FP", FP_
                    print "FN", FN_
                
                total_GT += len(GT_bboxes)
                total_pred += len(predicted_bboxes)
                TP += TP_
                FP += FP_
                FN += FN_

                n_prev_classes = sum(n_classes[:dataset]) - (len(n_classes[:dataset]) - 1) - 1
                for pos in range(len(TP_c)):
                    if(pos == 0):
                        TP_classes[pos] += TP_c[pos]
                        FP_classes[pos] += FP_c[pos]
                        FN_classes[pos] += FN_c[pos]
                    else:
                        TP_classes[pos+n_prev_classes-1] += TP_c[pos]
                        FP_classes[pos+n_prev_classes-1] += FP_c[pos]
                        FN_classes[pos+n_prev_classes-1] += FN_c[pos]

                for y in GT_Y:
                    if(y == 0):
                        total_GT_classes[y] += 1
                    else:
                        total_GT_classes[y+n_prev_classes] += 1
                for y in predicted_Y:
                    if(y == 0):
                        total_pred_classes[y] += 1
                    else:
                        total_pred_classes[y+n_prev_classes] += 1
                
        
    
        # Compute general precision / recall / accuracy measures
        if((TP+FP) == 0):
            precision = 0.0
        else:
            precision = float(TP)/(TP+FP)
        if((TP+FN) == 0):
            recall = 0.0
        else:
            recall = float(TP)/(TP+FN)
        if((FP+FN+TP) == 0):
            accuracy = 0.0
        else:
            accuracy = float(TP)/(FP+FN+TP)


        # Compute class-specific precision - recall
        precision_classes = np.zeros((n_tot))
        recall_classes = np.zeros((n_tot))
        accuracy_classes = np.zeros((n_tot))
        for c in range(n_tot):
            TP = TP_classes[c]
            FP = FP_classes[c]
            FN = FN_classes[c]
            if((TP+FP) == 0):
                precision_classes[c] = 0.0
            else:
                precision_classes[c] = float(TP)/(TP+FP)
            if((TP+FN) == 0):
                recall_classes[c] = 0.0
            else:
                recall_classes[c] = float(TP)/(TP+FN)
            if((FP+FN+TP) == 0):
                accuracy_classes[c] = 0.0
            else:
                accuracy_classes[c] = float(TP)/(FP+FN+TP)
        
        # store results
        general_measures[thres] = [precision, recall, accuracy, total_GT, total_pred]
        class_measures[thres] = [precision_classes, recall_classes, accuracy_classes, total_GT_classes, total_pred_classes]
    
    return [general_measures, class_measures]   
    

def computeMeasures(IoU, n_classes, predicted_bboxes, predicted_Y, predicted_scores, GT_bboxes, GT_Y):
    '''
        Computes TP, FP, and FN given a set of GT and Prediction BBoxes
    '''
    # General counters (without applying class distinctions)
    TP = 0
    FP = 0
    FN = 0
    
    # Class-specific counters
    TP_classes = np.zeros((n_classes))
    FP_classes = np.zeros((n_classes))
    FN_classes = np.zeros((n_classes))
    
    if(len(predicted_Y) > 0):
        using_recognition = True
    else:
        using_recognition = False
    
    # Compute IoU for each pair of bounding boxes in (GT, pred)
    iou_values = []
    pred_ids = []
    match_bboxes = []
    for i, gt in enumerate(GT_bboxes):
        for j, pred in enumerate(predicted_bboxes):
            # compute IoU
            iou_values.append(computeIoU(gt, pred))
            pred_ids.append(j)
            match_bboxes.append([i, j])
    
    # Compute matchings (sorted by IoU)
    final_matches = [] # stores the final indices for [gt,pred] matches
    matched_gt = [False for i in range(len(GT_bboxes))]
    matched_pred = [False for i in range(len(predicted_bboxes))]
    #max_iou = np.argsort(np.array(iou_values, dtype=np.float))[::-1]
    max_scores = np.argsort(np.array(predicted_scores, dtype=np.float))[::-1]
    
    # Sort predictions by "scores"
    i = 0
    while(i < len(max_scores) and not all(matched_gt)):
        #m = match_bboxes[max_iou[i]]
        this_pred_id = max_scores[i]
        m_list = [[p_, match_bboxes[p_]] for p_,p in enumerate(pred_ids) if p==this_pred_id]
        this_iou = [iou_values[p] for p,m in m_list]
        max_iou = np.argsort(np.array(this_iou, dtype=np.float))[::-1]
        
        # Sort GT by IoU
        j = 0
        while(j < len(max_iou) and not matched_pred[this_pred_id]): # if pred has not been matched yet
            j_ind = max_iou[j]
            if(this_iou[j_ind] >= IoU and not matched_gt[m_list[j_ind][1][0]]):
                # Assign match
                matched_gt[m_list[j_ind][1][0]] = True
                matched_pred[this_pred_id] = True
                final_matches.append(m_list[j_ind][1])
            j += 1
        i += 1
        
    # Compute FPs, FNs and TPs on the current image
    for m in matched_gt:
        if(m):
            TP += 1
        else:
            FN += 1
    for m in matched_pred:
        if(not m):
            FP += 1

    # Compute class-specific counters
    if(using_recognition):
        # Check matching pairs
        for m in final_matches:
            y_gt = GT_Y[m[0]]
            y_pred = predicted_Y[m[1]]
            
            # GT and pred coincide
            if(y_gt == y_pred):
                TP_classes[y_gt] += 1
            # Missclassified but correctly localized
            else:
                FN_classes[y_gt] += 1
                FP_classes[y_pred] += 1
        # Check missed GT bboxes
        for i, m in enumerate(matched_gt):
            if(not m):
                FN_classes[GT_Y[i]] += 1
        # Check mislocalized Pred bboxes
        for i, m in enumerate(matched_pred):
            if(not m):
                FP_classes[predicted_Y[i]] += 1


    return [TP, FP, FN, TP_classes, FP_classes, FN_classes]
   

################################################
#
#    Auxiliary functions
#
################################################

def loadUECDetectionTest(ds, path_imgs, test_list):
    
    ds.path = ''
    
    # Get list of images
    all_ids = [] # all unique image ids
    list_imgs = [] # all unique images
    original_sizes = [] # sizes from original images
    
    match_ids = [] # indices to matching samples
    classes = [] # classes from each bounding box
    boxes = [] # bounding boxes
    
    logging.info('Reading detection '+test_list)
    i = 0
    with open(path_imgs+'/'+test_list, 'r') as list_:
        for line in list_:
            if(i > 0):
                line = line.rstrip('\n')
                line = line.split(' ')
                
                id = line[0].split('/')[-1].split('.')[0]
                try:
                    posid = all_ids.index(id)
                except:
                    posid = None
                if(not posid):
                    all_ids.append(id)
                    list_imgs.append(line[0])
                    match_ids.append(len(all_ids)-1)
                    original_sizes.append(misc.imread(line[0]).shape)
                else:
                    match_ids.append(posid)
                    
                boxes.append([float(line[2]), float(line[3]), float(line[4]), float(line[5])])
                classes.append(int(line[1]))
            i += 1
            
            if(i % 1000 == 0):
                logging.info('Read '+str(i)+' lines...')
        logging.info('Read '+str(i)+' lines...')
            
    ds.setList(list_imgs, 'test')
    
    samples_detection = dict()
    samples_detection['all_ids'] = all_ids
    samples_detection['list_imgs'] = copy.copy(list_imgs)
    samples_detection['match_ids'] = match_ids
    samples_detection['classes'] = classes
    samples_detection['boxes'] = boxes
    samples_detection['original_sizes'] = original_sizes
    return [ds, samples_detection]

    
def loadFoodVsNoFoodDataset(size_crop=[224, 224, 3]):
    """
        Loads the Food Vs NoFood dataset
    """
    logging.info('Reading dataset info.')
    
    path_general = '/media/HDD_2TB/marc/FoodCNN_Data/data_split'
    val_file = 'val.txt'
    train_file = 'train.txt'

    # Create temporal train and val split files
    val_names = []
    val_labels = []
    train_names = []
    train_labels = []
    with open(path_general + '/' + val_file, 'r') as list_:
        for line in list_:
            line = line.rstrip('\n')
            im_name, label = line.split(' ')
            val_labels.append(label)
            val_names.append(im_name)
    
    with open(path_general + '/tmp_' + val_file, 'w') as the_file:
        for line in val_names:
            the_file.write('foodCNN_val/'+line+'\n')
            
    with open(path_general + '/' + train_file, 'r') as list_:
        for line in list_:
            line = line.rstrip('\n')
            im_name, label = line.split(' ')
            train_labels.append(label)
            train_names.append(im_name)
    
    with open(path_general + '/tmp_' + train_file, 'w') as the_file:
        for line in train_names:
            the_file.write('foodCNN_train/'+line+'\n')
    
    
    ds = Dataset('FoodVsNoFood', path_general)
    ds.setImageSizeCrop(size_crop)
    ds.setClasses('/media/HDD_2TB/marc/FoodCNN_Data/data_split/classes.txt')
    ds.setList(path_general + '/tmp_' + val_file, 'val')
    ds.setList(path_general + '/tmp_' + train_file, 'train')
    
    # Load labels
    ds.setLabels(train_labels, 'train')
    ds.setLabels(val_labels, 'val')
    
    # Remove temporal images list
    os.remove(path_general + '/tmp_' + val_file)
    os.remove(path_general + '/tmp_' + train_file)
    
    logging.info(ds.name +' loaded.')
    
    return ds
    
    
    
def loadGoogleNet():
    
    logging.info('Loading GoogLeNet...')
    load_path = '/media/HDD_2TB/CNN_MODELS/GoogleNet'
    
    
    # Build model (loading the previously converted Caffe's model)
    googLeNet = Stage(2, 2, [224, 224, 3], [2], type='GoogleNet', model_name='GoogleNet-FoodVsNoFood',
                    structure_path=load_path+'/Keras_model_structure.json', 
                    weights_path=load_path+'/Keras_model_weights.h5')
    
    return googLeNet
  


def generateGAPforGoogleNet(net, new_model_name, learning_rate):
    # Remove all last layers (from 'pool4' included)
    layers_to_delete = ['pool4/3x3_s2_zeropadding', 'pool4/3x3_s2', 
                    'inception_5a/1x1','inception_5a/relu_1x1','inception_5a/3x3_reduce','inception_5a/relu_3x3_reduce',
                    'inception_5a/3x3_zeropadding','inception_5a/3x3','inception_5a/relu_3x3','inception_5a/5x5_reduce',
                    'inception_5a/relu_5x5_reduce','inception_5a/5x5_zeropadding','inception_5a/5x5','inception_5a/relu_5x5',
                    'inception_5a/pool_zeropadding','inception_5a/pool','inception_5a/pool_proj','inception_5a/relu_pool_proj',
                    'inception_5a/output','inception_5b/1x1','inception_5b/relu_1x1','inception_5b/3x3_reduce','inception_5b/relu_3x3_reduce',
                    'inception_5b/3x3_zeropadding','inception_5b/3x3','inception_5b/relu_3x3','inception_5b/5x5_reduce',
                    'inception_5b/relu_5x5_reduce','inception_5b/5x5_zeropadding','inception_5b/5x5','inception_5b/relu_5x5',
                    'inception_5b/pool_zeropadding','inception_5b/pool','inception_5b/pool_proj','inception_5b/relu_pool_proj',
                    'inception_5b/output','pool5/7x7_s1','pool5/drop_7x7_s1','loss3/classifier_flatten',
                    'loss3/classifier_food_vs_nofood']
    net.removeLayers(layers_to_delete)

    # Remove softmax output
    net.removeOutputs(['loss3/loss3'])
    
    # Add output
    net.model.add_output(name='inception_4e', input='inception_4e/output')
    net.setOptimizer()

    # Create GAP net
    gap_net = Stage(nInput=2, nOutput=2, input_shape=[14, 14, 832], output_shape=[2], type='GAP')
    gap_net.setOptimizer(lr=learning_rate)
        
    # Insert into staged network
    snet = Staged_Network(model_name=new_model_name)
    snet.addStage(net, out_name='inception_4e', training_is_enabled=False)
    snet.addStage(gap_net, in_name='inception_4e', out_name='GAP/softmax', balanced=True)

    return snet




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    main()
    
        