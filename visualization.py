from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.use('Agg') # run matplotlib without X server (GUI)
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
from skimage.transform import resize

cmap = plt.get_cmap('jet') # set colormap

# Function for plotting image and result obtained
def plot_localization(img_path_list, bboxes_list,
                      scores_list=None, img_FAMs_list=None, labels_list=None, classes_list=None, 
                      per_row=3, height=4, width=4):

    if not isinstance(img_path_list, list):
        img_path_list = [img_path_list]
        bboxes_list = [bboxes_list]
    if not isinstance(scores_list, list):
        scores_list = [scores_list]
    if not isinstance(labels_list, list):
        labels_list = [labels_list]
    if not isinstance(img_FAMs_list, list):
        img_FAMs_list = [img_FAMs_list]
    if not isinstance(classes_list, list):
        img_FAMs_list = [classes_list]
    
    if img_FAMs_list[0] is not None:
        num_images = len(img_path_list)*2
        FAMs=True
    else:
        num_images = len(img_path_list)
        FAMs=False
        
    num_rows = int(np.ceil(num_images/float(per_row)))
    f = plt.figure(1)
    f.set_size_inches(height*num_rows, width*per_row)
    
    for j, (img_path, bboxes) in enumerate(zip(img_path_list, bboxes_list)):
        
        if FAMs:
            next_idx = (j*2)+1
        else:
            next_idx = j+1
        
        im = misc.imread(img_path)
        ax = plt.subplot(num_rows,per_row,next_idx)
        plt.imshow(im)
        
        # remove axis
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        if FAMs:
            #im_fam = misc.imread(img_FAMs_list[j])
            im_fam = np.load(img_FAMs_list[j])
            
            norm = (im_fam - np.min(im_fam)) / (np.max(im_fam) - np.min(im_fam)) # 0-1 normalization
            heat = cmap(norm) # apply colormap
            heat = np.delete(heat, 3, 2) # ???
            heat = np.array(heat[:, :, :3]*255, dtype=np.uint8)
            heat = np.array(resize(heat, im.shape[:2], order=1, preserve_range=True), dtype=np.uint8)
            im_fam = np.array(im*0.3 + heat*0.7, dtype=np.uint8)            
            ax_fam = plt.subplot(num_rows,per_row,next_idx+1)
            plt.imshow(im_fam)
        
        # remove axis
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        for i in range(len(bboxes)):
            box = bboxes[i]
            if scores_list[0] is not None:
                plot_text = str(scores_list[j][i])
            else:
                plot_text = ''
            if labels_list[0] is not None:
                if classes_list is not None:
                    plot_text = str(classes_list[labels_list[j][i]]) + ' (' + plot_text + ')'
                else:
                    plot_text = str(labels_list[j][i]) + ' (' + plot_text + ')'


            ax.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                   facecolor="none", edgecolor='blue', linewidth=3.0))
            ax.text(box[0]+20, box[1]+50, plot_text, backgroundcolor='cornflowerblue')

    return f
