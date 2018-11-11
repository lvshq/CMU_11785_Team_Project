import os
import numpy as np
from PIL import Image

IMG_SIZE= (150, 150)
TRAIN_IMGS_PATH = "./Data/train_imgs_v0/"
TEST_IMGS_PATH = "./Data/test_imgs_v0"

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
    im= im.resize(IMG_SIZE, Image.ANTIALIAS)
    return im


def sample_zero_mean(x):
    return x - np.mean(x, axis=1, keepdims=True)


def gcn(x, scale=55., bias=0.01):
    return scale * x / np.sqrt(bias + np.var(x, axis=1, keepdims=True))


def preprocess(x):
    """
    Flatten train data and make each sample have a mean of zero.
    Also GCN each sample (assume sample mean=0).
    Finally restore original shape.
    """
    sample_num = x.shape[0]
    original_shape = x.shape
    x = np.reshape(x, (sample_num, -1))
    x = sample_zero_mean(x)
    x = gcn(x)
    x = np.reshape(x, original_shape)
    return x


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


def get_label(fname):
    """
    The image filename is in format: label_xxx.jpg, 
    where xxx is the number of this image in such label.
    """
    label = "".join(fname.split('_')[:-1])
    return label


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
            label = get_label(fname)
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
            label = get_label(fname)
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


label_dict = {}
idx = 0
for fname in os.listdir(TRAIN_IMGS_PATH):
    label = get_label(fname)
    if label not in label_dict:
        label_dict[label] = idx
        idx += 1
# Ensure test data's label is same with train data
for fname in os.listdir(TEST_IMGS_PATH):
    label = get_label(fname)
    if label not in label_dict:
        raise Exception("Invalid label: %s" % label)

#test
for label in label_dict:
    print(label)

