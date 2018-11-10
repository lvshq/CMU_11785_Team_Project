import os
import numpy as np
from PIL import Image

def crop_and_scale_image(im):
    """ Crops and scales a given image. 
        Args: 
            im (PIL Image) : image object to be cropped and scaled
        Returns: 
            (PIL Image) : cropped and scaled image object
    """
    width,height = im.size
    if width > height:
        diff = width - height
        box = diff/2, 0, width - (diff - diff/2), height
    else:
        diff = height - width
        box = 0, diff/2, width, height - (diff - diff/2)
    im = im.crop(box)
    toSize = 150,150
    im= im.resize(toSize, Image.ANTIALIAS)
    return im

def fname_to_network_input(fname):
    """ Creates the input for a VGG network from the filename 
        Args: 
            fname (string) : the filename to be parsed
        Returns: 
            (numpy ndarray) : the array to be passed into the VGG network as a single example
    """
    im = Image.open(fname)
    im = crop_and_scale_image(im)
    if im.mode is not 'RGB':
        im = im.convert('RGB')
    npim = np.asarray(im)
    npim = np.rollaxis(npim, 2)
    return npim

label_dict = {}
idx = 0
for fname in os.listdir("../existing material/Data/train_imgs/"):
    label = fname.split('_')[0]
    if label not in label_dict:
        label_dict[label] = idx
        idx += 1

def load_data(path):
    """  Load training data and test data
         Args:
            path (string) : the data path
         Returns:
            data, labels (numpy array)
    """
    dname = path
    im_paths = []
    labels = []
    for fname in os.listdir(dname):
        if fname.endswith('.jpg'):
            label = fname.split('_')[0]
            if label not in label_dict:
                print('wrong label ', label)
            im_paths.append(dname+fname)
            try:
                labels.append(label_dict[label])
            except:
                continue 
    data = []
    for file in im_paths:
        data.append(fname_to_network_input(file))
        
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels

