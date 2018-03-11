import glob
import os
import tables
# import internetarchive
import numpy as np

from subprocess import call
from scipy.misc import imresize
from scipy.ndimage import zoom
from PIL import Image
from qtim_tools.qtim_utilities.file_util import grab_files_recursive

def internet_archive_login():

    return

def internet_archive_download(destination_directory='E:\Pages', collection='MBLWHOI'):

    for i in internetarchive.search_items('collection:' + collection):
        archive_id = i['identifier']
        try:
            if not os.path.exists(os.path.join(destination_directory, archive_id)):
                internetarchive.download(archive_id, verbose=True, glob_pattern='*.pdf', destdir=destination_directory)
            elif os.listdir(os.path.join(destination_directory, archive_id)) == []:
                internetarchive.download(archive_id, verbose=True, glob_pattern='*.pdf', destdir=destination_directory)
        except KeyboardInterrupt:
            raise
        except:
            print 'ERROR downloading', archive_id
    return

def convert_pdf_to_image(conversion_directory='E:\Pages', output_directory='E:\Pages_Images', ghostscript_path='"C:/Program Files/gs/gs9.22/bin/gswin64c.exe"'):

    documents = glob.glob(os.path.join(conversion_directory, '*/'))

    for document in documents:
        pdfs = glob.glob(os.path.join(document, '*.pdf'))
        document_basename = os.path.join(output_directory, os.path.basename(os.path.dirname(document)))

        if os.path.exists(document_basename + '-1.png'):
            print 'Skipping', document_basename
            continue

        for pdf in pdfs:

            if pdf.endswith('_bw.pdf'):
                continue

            command = ghostscript_path + " -dBATCH -dNOPAUSE -sDEVICE=png16m -r144 -sOutputFile=" + document_basename + "-%d.png" + ' ' + pdf
            print(command)
            call(command, shell=True)

    return

def preprocess_image(input_directory='E:/Pages_Images', output_directory='E:/Pages_Images_Preprocessed', resize_shape=(512, 512), verbose=True):

    print input_directory
    images = glob.glob(os.path.join(input_directory, '*.*'))

    for filepath in images:

        try:

            output_filepath = os.path.join(output_directory, os.path.basename(filepath))
            if not os.path.exists(output_filepath):
                if verbose:
                    print 'Processing...', filepath

                img = Image.open(filepath)
                data = np.asarray(img, dtype='uint8')

                print data.shape

                data = imresize(data, resize_shape)

                img = Image.fromarray(data)
                img.save(output_filepath)

        except KeyboardInterrupt:
            raise
        except:
            print 'ERROR converting', filepath

    return

def create_hdf5_file(output_filepath, num_cases, image_shape=(64, 64)):

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')

    data_shape = (0,) + image_shape + (3,)

    hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)
    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    return hdf5_file


def store_to_hdf5(data_directory, hdf5_file, image_shape, verbose=True):

    input_images = glob.glob(os.path.join(data_directory, '*.png'))

    hdf5_file = create_hdf5_file(hdf5_file, num_cases=len(input_images), image_shape=image_shape)

    for image in input_images:
        try:
            if verbose:
                print(image)
            img = Image.open(image)
            data = np.asarray(img)
            hdf5_file.root.data.append(data[np.newaxis])
            hdf5_file.root.imagenames.append(np.array(os.path.basename(image))[np.newaxis][np.newaxis])
        except:
            print 'ERROR WRITING TO HDF5', image

    return hdf5_file

def store_preloaded_hdf5_file(input_directory, output_filepath, output_directory=None, verbose=True, mask_directory=None, dimensions=[4,8,16,32,64,128,256,512], rop=False):

    images = glob.glob(os.path.join(input_directory, '*.*'))
    num_cases = len(images)

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    if mask_directory is not None:
        channels = 4
    else:
        channels = 3

    for dimension in dimensions:
        data_shape = (0, dimension, dimension, channels)
        hdf5_file.create_earray(hdf5_file.root, 'data_' + str(dimension), tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    for idx, filepath in enumerate(images):

        hdf5_file.root.imagenames.append(np.array(os.path.basename(filepath))[np.newaxis][np.newaxis])

        img = Image.open(filepath)
        data = np.asarray(img, dtype=float)[:,:,0:3]
        print data.shape

        if rop:
            if data.shape != (480, 640, 3):
                data = imresize(data, (480, 640))

        if mask_directory is not None:
            try:
                img_mask = Image.open(os.path.join(mask_directory, os.path.basename(filepath)[0:-3] + 'png'))
            except:
                continue
            data_mask = np.asarray(img_mask, dtype=float)

        for dimension in dimensions:
            try:
            # if True:

                if verbose:
                    print 'Processing...', os.path.basename(filepath), 'at', dimension, ', idx', idx

                resized_data = imresize(data, (dimension, dimension))

                if mask_directory is not None:
                    resized_mask = imresize(data_mask, (dimension, dimension))[:,:,np.newaxis]
                    resized_data = np.concatenate((resized_data, resized_mask), axis=2)

                getattr(hdf5_file.root, 'data_' + str(dimension)).append(resized_data[np.newaxis])

                if output_directory is not None:
                    output_filepath = os.path.join(output_directory, str(dimension) + '_' + os.path.basename(filepath))
                    if not os.path.exists(output_filepath):
                        img = Image.fromarray(data)
                        img.save(output_filepath)

            except KeyboardInterrupt:
                raise
            except:
                print 'ERROR converting', filepath, 'at dimension', dimension
 
    hdf5_file.close()

    return

def store_test_rop_hdf5(input_directory, output_filepath, output_directory=None, verbose=True, mask_directory=None, dimensions=[4,8,16,32,64,128,256,512], classes=['No', 'Pre-Plus', 'Plus'], category_num=2):

    images = grab_files_recursive(input_directory)
    num_cases = len(images)

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')


    if mask_directory is not None:
        channels = 4
    else:
        channels = 3

    for class_id in classes:
        hdf5_file.create_earray(hdf5_file.root, 'imagenames_' + class_id, tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)
        hdf5_file.create_earray(hdf5_file.root, 'classifications_' + class_id, tables.Float32Atom(), shape=(0,category_num), filters=filters, expectedrows=num_cases)
        for dimension in dimensions:
            data_shape = (0, dimension, dimension, channels)
            hdf5_file.create_earray(hdf5_file.root, 'data_' + class_id + '_' + str(dimension), tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    class_map = {'No': 0, 'Pre-Plus': 1, "Plus": 2, "os": 1, "od": 2}

    for idx, filepath in enumerate(images):

        split_file = str.split(filepath, '_')
        eye_side = split_file[-2]
        classification = split_file[-1][:-4]

        img = Image.open(filepath)
        data = np.asarray(img, dtype=float)[:,:,0:3]
        print data.shape

        if data.shape != (480, 640, 3):
            data = imresize(data, (480, 640))

        if mask_directory is not None:
            try:
                img_mask = Image.open(os.path.join(mask_directory, os.path.basename(filepath)[0:-3] + 'png'))
            except:
                continue
            data_mask = np.asarray(img_mask, dtype=float)

        getattr(hdf5_file.root, 'imagenames_' + classification).append(np.array(os.path.basename(filepath))[np.newaxis][np.newaxis])
        getattr(hdf5_file.root, 'classifications_' + classification).append(np.array([class_map[eye_side], class_map[classification]])[np.newaxis])

        for dimension in dimensions:
            # try:
            if True:

                if verbose:
                    print 'Processing...', os.path.basename(filepath), 'at', dimension, ', idx', idx

                resized_data = imresize(data, (dimension, dimension))

                if mask_directory is not None:
                    resized_mask = imresize(data_mask, (dimension, dimension))[:,:,np.newaxis]
                    resized_data = np.concatenate((resized_data, resized_mask), axis=2)


                print resized_data.shape
                getattr(hdf5_file.root, 'data_' + classification + '_' + str(dimension)).append(resized_data[np.newaxis])

                if output_directory is not None:
                    output_filepath = os.path.join(output_directory, str(dimension) + '_' + os.path.basename(filepath))
                    if not os.path.exists(output_filepath):
                        img = Image.fromarray(data)
                        img.save(output_filepath)

            # except KeyboardInterrupt:
                # raise
            # except:
                # print 'ERROR converting', filepath, 'at dimension', dimension
 
    hdf5_file.close()

    return

class PageData(object):

    def __init__(self, collection='MBLWHOI', shape=(64,64), hdf5=None, preloaded=False, preprocessed=False, classes=['Plus', 'Plus', 'Plus'], channel=None):

        self.collection = collection
        self.shape = shape
        self.hdf5 = hdf5
        self.preloaded = preloaded
        self.preprocessed = preprocessed
        self.classes = classes
        self.channel = channel

        if classes is None:
            self.image_num = getattr(self.hdf5.root, 'data_4').shape[0]
            self.indexes = np.arange(self.image_num)
            np.random.shuffle(self.indexes)
        else:
            self.image_num = []
            self.indexes = []
            for class_id in classes:
                self.image_num += [getattr(self.hdf5.root, 'data_' + class_id + '_4').shape[0]]
                self.indexes += [np.arange(self.image_num[-1])]

        self.zoom_mapping = {9:'1024', 8:'512', 7:'256', 6:'128', 5:'64', 4:'32', 3:'16', 2:'8', 1:'4'}

    def get_next_batch(self, batch_num=0, batch_size=64, zoom_level=1, mode='preloaded'):

        total_batches = self.image_num / batch_size - 1

        if batch_num % total_batches == 0:
            np.random.shuffle(self.indexes)

        indexes = self.indexes[(batch_num % total_batches) * batch_size: (batch_num % total_batches + 1) * batch_size]

        if self.preloaded:
            if self.preprocessed:
                return np.array([getattr(self.hdf5.root, 'data_' + str(self.zoom_mapping[zoom_level]))[idx] for idx in indexes]), None
            else:
                data = np.array([getattr(self.hdf5.root, 'data_' + str(self.zoom_mapping[zoom_level]))[idx] for idx in indexes])
                # data[...,-1] = data[...,-1]
                data = data / 127.5 - 1
                return data, None

        else:
            data = np.array([self.hdf5.root.data[idx] for idx in indexes]) / 127.5 - 1

            if zoom_level == 1:
                return data
            else:
                data = zoom(data, zoom=[1,1.0/zoom_level,1.0/zoom_level,1])
                return data

    def get_next_batch_classed(self, batch_num=0, batch_size=64, zoom_level=1, mode='preloaded', infogan=False):

        # Gotta be some of the dumbest code I've ever written in this function.
        batch_sizes = [batch_size / len(self.classes) for i in xrange(len(self.classes) - 1)] 
        batch_sizes += [batch_size - np.sum(batch_sizes)]

        data = []
        classes = []

        for class_idx, class_id in enumerate(self.classes):

            total_batches = self.image_num[class_idx] / batch_sizes[class_idx] - 1

            if batch_num % total_batches == 0:
                np.random.shuffle(self.indexes[class_idx])

            if self.channel is None:
                data += [getattr(self.hdf5.root, 'data_' + class_id + '_' + str(self.zoom_mapping[zoom_level]))[idx] for idx in self.indexes[class_idx][(batch_num % total_batches) * batch_sizes[class_idx]:(batch_num % total_batches + 1) * batch_sizes[class_idx]]]
            else:
                data += [getattr(self.hdf5.root, 'data_' + class_id + '_' + str(self.zoom_mapping[zoom_level]))[idx,...,self.channel][...,np.newaxis] for idx in self.indexes[class_idx][(batch_num % total_batches) * batch_sizes[class_idx]:(batch_num % total_batches + 1) * batch_sizes[class_idx]]]

            classes += [getattr(self.hdf5.root, 'classifications_' + class_id)[idx] for idx in self.indexes[class_idx][(batch_num % total_batches) * batch_sizes[class_idx]:(batch_num % total_batches + 1) * batch_sizes[class_idx]]]


        if self.preloaded:
            if self.preprocessed:
                data = np.array(data)
            else:
                data = np.array(data) / 127.5 - 1
            if infogan:
                return data, classes
            else:
                return data, None




    def save_test_image(self):

        data = getattr(self.hdf5.root, 'data_512')[0,:,:,:3]
        print data.shape
        print np.unique(data)
        import scipy
        scipy.misc.imsave('test_image.png', data)

    def close(self):

        self.hdf5.close()

if __name__ == '__main__':

    pass