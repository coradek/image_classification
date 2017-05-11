import os
import re
import cv2
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import cPickle as pickle

"""Credit: KERNIX blog - Image classification
    with a pre-trained deep neural network"""

def create_graph():
    # (confirm) creates .pb graph file in imagenet directory
    model_dir = 'imagenet'
    with gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(image_list):
    """
    INPUT: a list of photo file paths.
    OUTPUT: numpy array of feature vectors
    """

    print("commencing feature extraction")
    nb_features = 2048
    features = np.empty((len(image_list), nb_features))
    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(image_list):
            if ind %10 ==0:
                print("processing image {}: {}".format(ind,image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()

            predictions = sess.run(next_to_last_tensor,
                                    {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
    print("feature extraction complete")
    return features


def create_validation_set(root, num=100):
    """
    INPUT: path to project location
    OUTPUT: test_train and validation directories create
            image files moved
    """
    #TODO: check if '/test_images' exists
    # if not create '/test_images' and move images from training
    # else pass

    # root = "/Users/ophidian/coradek/WorkSpace/AeyesafeMVP/"
    image_dir = "data/test_train_images/"
    test_dir = "data/validation_images/"
    path = os.path.join(root, image_dir)
    dest = os.path.join(root, test_dir)

    if not os.path.exists(dest):
        os.mkdir(dest)
        labels = []
        paths = []
        for p, dirs, files in os.walk(path):
            for dd in dirs:
                # get the last 100 images in each class
                # (consider also get random 100 images from each class)
                for n, im in enumerate(os.listdir(os.path.join(p,dd))[-num:]):
                    if im[-4:].lower() == '.jpg':
                        current = os.path.join(p, dd, im)
                        new_name = "{}_{}.jpg".format(dd,n+1)
                        final = os.path.join(dest,new_name)
                        labels.append(dd)
                        paths.append(final)
                        os.rename(current, final)
        return paths, labels

    else:
        print "'test_train_images' and 'validation_images' \
                directories already exist"


def create_validation_df(directory):
    """
    INPUT: path to '/validation_images'
    OUTPUT: data frame with 'image_path', 'label', and 2048 tensorflow feature columns
    """
    # directory = "/Users/ophidian/coradek/WorkSpace/AeyesafeMVP/data/validation_images"
    file_list = os.listdir(directory)
    paths = []
    labels = []
    for item in file_list:
        if item[-4:].lower() == '.jpg':
            paths.append(os.path.join(directory, item))
            labels.append(item[:-7])

    features = extract_features(paths)
    df = pd.DataFrame(features)
    df['label'] = labels
    df['file_path'] = paths

    return df


def convert_to_jpg(path):
    """
    Convert images to .jpg for use in inception-v3
    """
    if path[:-4] is not '.jpg':
        new_name = ''.join(path.split('.')[:-1]) + '.jpg'
        im =cv2.imread(path)
        cv2.imwrite(new_name, im)
        return new_name


def get_paths(directory):
    """
    INPUT: directory of photos
    Converts ".png" and ".bmp" to ".jpg"
    OUTPUT: list of ".jpg" file paths
    """
    image_path_list = []
    for p, dirs, files in os.walk(directory):
        for ff in files:
            if ff[-4:].lower() == '.jpg':
                file_path = os.path.join(p,ff)
                image_path_list.append(file_path)
            elif (ff[-4:].lower() == '.png') or (ff[-4:].lower() == '.bmp'):
                file_path = os.path.join(p,ff)
                file_path = convert_to_jpg(file_path)
                image_path_list.append(file_path)
            else:
                print("file - {} - not processed.\
                 \nnon-image file or format not supported".format(ff))
    return image_path_list


def get_labels(image_list):
    """
    INPUT: list of '.jpg' file paths where each class of images
           is in its own subdirectory
    OUTPUT: list of labels
    """
    labels = []
    for item in image_list:
        labels.append(item.split(os.path.sep)[-2])
    return labels


def create_df(directory):
    """
    INPUT: directory containing subdirectories named according to the
           class label of the contained images
    OUTPUT: data frame with 'image_path', 'label', and 2048 feature columns
    """
    print("creating dataframe")
    path_list = get_paths(directory)
    labels = get_labels(path_list)
    features = extract_features(path_list)
    df = pd.DataFrame(features)
    df['label'] = labels
    df['file_path'] = path_list
    print("dataframe complete")
    return df


def main():
    # TODO:
    # inform of process, ask before proceeding
    # separate validation data
    # create and save training and validation dataframes
    pass


if __name__ == '__main__':
    main()
