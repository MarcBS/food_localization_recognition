
import random
import itertools
import os
import shutil

import numpy as np
from scipy import misc


''' This script is used to create a training/validation/test split of the data stored in the UECFood-256 dataset format '''

def main():
    
    #################################################
    # Parameters ( IMPORTANT!: do NOT use spaces in paths or in image names! )
    path_datasets = '/media/HDD_2TB/DATASETS'
    path = path_datasets+'/UECFOOD256';
    categories = 'category.txt'
    bbox_files = 'bb_info.txt'

    imgs_format = 'jpg'

    split = [0.7, 0.1, 0.2] # [train, val, test] percentages

    files_split = ['train_list.txt', 'val_list.txt', 'test_list.txt']

    # Localization result
    generate_localization = True
    path_localization = path

    # Recognition result
    generate_recognition = True
    path_recognition = path_datasets+'/UECFOOD256_recognition'
    
    #################################################
    
    # Read categories ids and names
    print "Reading categories"
    category_ids = []
    category_names = []
    with open(path+'/'+categories, 'r') as list_:
        for i,line in enumerate(list_):
            if(i > 0): # skip header
                line = line.rstrip('\n')
                line = line.split('\t')
                category_ids.append(int(line[0]))
                category_names.append(line[1])
            
    
    # Read list of images from each category
    print "Reading images and their bboxes from each category"
    categories_images = []
    categories_bbox_info = []
    for id_pos, id in enumerate(category_ids):
        # Create empty list for images list and bbox info
        categories_images.append([])
        categories_bbox_info.append([])
        
        # Read file
        imgs_file_list = path+'/'+str(id)+'/'+bbox_files
        with open(imgs_file_list, 'r') as list_:
            for i,line in enumerate(list_):
                if(i > 0): # skip header
                    line = line.rstrip('\n')
                    line = line.split(' ')
                    categories_images[id_pos].append(line[0])
                    line = line[1:]
                    for j,l in enumerate(line):
                        line[j] = float(l)
                    categories_bbox_info[id_pos].append(line)
    
    
    # Split categories in train/val/test
    print "Splitting categories in train/val/test"
    train_list = [] # all lists store image ids (names)
    val_list = []
    test_list = []
    for id_pos, id in enumerate(category_ids):
        # Divide number of samples
        n_imgs = len(categories_images[id_pos])
        n_test = int(np.floor(n_imgs*split[2]))
        n_val = int(np.floor(n_imgs*split[1]))
        n_train = n_imgs - (n_test + n_val)
        
        # Shuffle samples
        shuffled = random.sample(categories_images[id_pos], n_imgs)
        
        train_list.append(shuffled[:n_train])
        val_list.append(shuffled[n_train:n_train+n_val])
        test_list.append(shuffled[n_train+n_val:])
    
    
    
    # Get train/val/test complete lists from any category
    print "Joining all train, all val and all test samples"
    all_train_list = list(np.unique(list(itertools.chain(*train_list))))
    all_val_list = list(np.unique(list(itertools.chain(*val_list))))
    all_test_list = list(np.unique(list(itertools.chain(*test_list))))
    
    
    # Solve splits overlaps (giving preference in the order val(1), test(2), train(3))
    print "Solving splits overlaps"
    i = 0
    while(i < len(all_train_list)): # give priority to val and test over train
        if(all_train_list[i] in all_val_list): # training sample is in val set too
            all_train_list.pop(i)
        elif(all_train_list[i] in all_test_list): # training sample is in test set too
            all_train_list.pop(i)    
        else:
            i += 1
            
    i = 0
    while(i < len(all_test_list)): # give priority to val over test
        if(all_test_list[i] in all_val_list): # test sample is in val set too
            all_test_list.pop(i)
        else:
            i += 1
            
    
    
    #################################################
    #    LOCALIZATION
    #################################################    
    
    if(generate_localization):
        print "Writing localization split files"
        # Store localization split files
        if(not os.path.isdir(path_localization)):
            os.makedirs(path_localization)
        
        # Store file with classes
        file = open(path_localization +'/classes.txt', 'w')
        for c in category_names:
            file.write(c+'\n')
        file.close()
        
        ########### Training
        file = open(path_localization +'/'+ files_split[0], 'w')
        file.write('path_img category_id x1 y1 x2 y2\n')
        for img in all_train_list:
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img+'.'+imgs_format, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                path_img = occ[0]
                category_img = occ[1]
                bbox_img = str(occ[2])+' '+str(occ[3])+' '+str(occ[4])+' '+str(occ[5])
                file.write(path_img + ' ' + category_img + ' ' + bbox_img + '\n')
        file.close()
        
        ########### Val
        file = open(path_localization +'/'+ files_split[1], 'w')
        file.write('path_img category_id x1 y1 x2 y2\n')
        for img in all_val_list:
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img+'.'+imgs_format, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                path_img = occ[0]
                category_img = occ[1]
                bbox_img = str(occ[2])+' '+str(occ[3])+' '+str(occ[4])+' '+str(occ[5])
                file.write(path_img + ' ' + category_img + ' ' + bbox_img + '\n')
        file.close()
        
        ########### Test
        file = open(path_localization +'/'+ files_split[2], 'w')
        file.write('path_img category_id x1 y1 x2 y2\n')
        for img in all_test_list:
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img+'.'+imgs_format, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                path_img = occ[0]
                category_img = occ[1]
                bbox_img = str(occ[2])+' '+str(occ[3])+' '+str(occ[4])+' '+str(occ[5])
                file.write(path_img + ' ' + category_img + ' ' + bbox_img + '\n')
        file.close()
        
        
    
    #################################################
    #    RECOGNITION
    ################################################# 
    
    if(generate_recognition):
        print "Writing recognition split files"
        # Store recognition split files
        # Write necessary folders
        if(os.path.isdir(path_recognition)):
            shutil.rmtree(path_recognition)
        os.makedirs(path_recognition)
        for id in category_ids:
            os.makedirs(path_recognition+'/'+str(id))
         
        # Store file with classes
        file = open(path_recognition +'/classes.txt', 'w')
        for c in category_names:
            file.write(c+'\n')
        file.close()
            
        ########### Training
        file = open(path_recognition +'/'+ files_split[0], 'w')
        file.write('path_img category_id\n')
        for img in all_train_list:
            count_crops = 0
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Load image for extracting crops
            img_mat = misc.imread(occurrences[0][0] + '.' + imgs_format)
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                count_crops += 1
                path_img = path_recognition + '/' + occ[1] + '/' + img + '_'+ str(count_crops) + '.' + imgs_format
                category_img = occ[1]
                img_crop = img_mat[int(occ[3]):int(occ[5]), int(occ[2]):int(occ[4])]
                try:
                    misc.imsave(path_img, img_crop) # save crop
                except:
                    print "ERROR:"
                    print occurrences
                file.write(path_img + ' ' + category_img + '\n')
        file.close()
        
        ########### Val
        file = open(path_recognition +'/'+ files_split[1], 'w')
        file.write('path_img category_id\n')
        for img in all_val_list:
            count_crops = 0
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Load image for extracting crops
            img_mat = misc.imread(occurrences[0][0] + '.' + imgs_format)
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                count_crops += 1
                path_img = path_recognition + '/' + occ[1] + '/' + img + '_'+ str(count_crops) + '.' + imgs_format
                category_img = occ[1]
                img_crop = img_mat[int(occ[3]):int(occ[5]), int(occ[2]):int(occ[4])]
                try:
                    misc.imsave(path_img, img_crop) # save crop
                except:
                    print "ERROR:"
                    print occurrences
                file.write(path_img + ' ' + category_img + '\n')
        file.close()
        
        ########### Test
        file = open(path_recognition +'/'+ files_split[2], 'w')
        file.write('path_img category_id\n')
        for img in all_test_list:
            count_crops = 0
            # Find all occurrences of this image and their categories
            occurrences = []
            for id_pos, id in enumerate(category_ids):
                categories_images[id_pos]
                occ = [[path+'/'+str(id)+'/'+img, str(id)]+categories_bbox_info[id_pos][i] for i, elem in enumerate(categories_images[id_pos]) if elem==img]
                occurrences += occ
            
            # Load image for extracting crops
            img_mat = misc.imread(occurrences[0][0] + '.' + imgs_format)
            
            # Write in file each occurrence found for the current image
            for occ in occurrences:
                count_crops += 1
                path_img = path_recognition + '/' + occ[1] + '/' + img + '_'+ str(count_crops) + '.' + imgs_format
                category_img = occ[1]
                img_crop = img_mat[int(occ[3]):int(occ[5]), int(occ[2]):int(occ[4])]
                try:
                    misc.imsave(path_img, img_crop) # save crop
                except:
                    print "ERROR:"
                    print occurrences
                file.write(path_img + ' ' + category_img + '\n')
        file.close()
    
    
    print 'Done!'



def addNoFood():
    ''' Adds examples of NoFood to the recognition dataset '''

    #################################################
    # Parameters ( IMPORTANT!: do NOT use spaces in paths or in image names! )
    path_datasets = '/media/HDD_2TB/DATASETS'
    path_recognition = path_datasets+'/UECFOOD256_recognition'
    
    files_split = ['train_list.txt', 'val_list.txt', 'test_list.txt']
    
    # No Food parameters
    path_nofood = '/media/HDD_2TB/marc/FoodCNN_Data/data_split'
    folders_nofood = ['foodCNN_train', 'foodCNN_val']
    lists = ['train.txt', 'val.txt']
    samples_proportion = [82, 15, 27] # number of train, val and test samples chosen (corresponds to mean samples per classes)
    #################################################
    
    print "Selecting NoFood samples for adding to "+ path_recognition
    
    # Create folder for no_food samples
    result_path_nofood = path_recognition + '/0'
    if(os.path.isdir(result_path_nofood)):
        shutil.rmtree(result_path_nofood)
    os.makedirs(result_path_nofood)
    
    # Select training samples
    no_food_train = []
    with open(path_nofood+'/'+lists[0], 'r') as list_:
        for i,line in enumerate(list_):
            line = line.rstrip('\n')
            line = line.split(' ')
            if(line[1] == '0'):
                no_food_train.append(line[0])            
    no_food_train = random.sample(no_food_train, samples_proportion[0])
    
    
    # Select val and test samples
    no_food_val = []
    no_food_test = []
    with open(path_nofood+'/'+lists[1], 'r') as list_:
        for i,line in enumerate(list_):
            line = line.rstrip('\n')
            line = line.split(' ')
            if(line[1] == '0'):
                no_food_val.append(line[0])          
    no_food_val = random.sample(no_food_val, samples_proportion[1]+samples_proportion[2])
    no_food_test = no_food_val[samples_proportion[1]:]
    no_food_val = no_food_val[:samples_proportion[1]]
    
    
    # Store in recognition folder and in images list
    file = open(path_recognition +'/'+ files_split[0], 'a')
    for s in no_food_train:
        shutil.copyfile(path_nofood+'/'+folders_nofood[0]+'/'+s, result_path_nofood+'/'+s)
        file.write(result_path_nofood+'/'+s+' 0\n')
    file.close()
    
    file = open(path_recognition +'/'+ files_split[1], 'a')
    for s in no_food_val:
        shutil.copyfile(path_nofood+'/'+folders_nofood[1]+'/'+s, result_path_nofood+'/'+s)
        file.write(result_path_nofood+'/'+s+' 0\n')
    file.close()
    
    file = open(path_recognition +'/'+ files_split[2], 'a')
    for s in no_food_test:
        shutil.copyfile(path_nofood+'/'+folders_nofood[1]+'/'+s, result_path_nofood+'/'+s)
        file.write(result_path_nofood+'/'+s+' 0\n')
    file.close()
        
    print 'Done!'
        

main()
addNoFood()    
