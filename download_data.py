import glob
import os
import tables
import internetarchive
import numpy as np

from subprocess import call
from scipy.misc import imresize
from scipy.ndimage import zoom
from PIL import Image

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

def preprocess_image(input_directory='E:/Pages_Images', output_directory='E:/Pages_Images_Preprocessed', resize_shape=(64, 64), verbose=True):

    images = glob.glob(os.path.join(input_directory, '*.png'))

    for filepath in images:

        try:

            output_filepath = os.path.join(output_directory, os.path.basename(filepath))
            if not os.path.exists(output_filepath):
                if verbose:
                    print 'Processing...', filepath

                img = Image.open(filepath)
                data = np.asarray(img, dtype='uint8')

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

    hdf5_file.close()
    fd = dg

    return hdf5_file


class PageData(object):

    def __init__(self, collection='MBLWHOI', shape=(64,64), hdf5=None):

        self.collection = collection
        self.shape = shape
        self.hdf5 = hdf5

        self.image_num = hdf5.root.data.shape[0]
        self.indexes = np.arange(self.image_num)
        np.random.shuffle(self.indexes)


    def get_next_batch(self, batch_num=0, batch_size=64, zoom_level=1):

        total_batches = self.image_num / batch_size - 1

        if batch_num % total_batches == 0:
            np.random.shuffle(self.indexes)

        indexes = self.indexes[(batch_num % total_batches) * batch_size: (batch_num % total_batches + 1) * batch_size]
        data = np.array([self.hdf5.root.data[idx] for idx in indexes]) / 127.5 - 1

        if zoom_level == 1:
            return data
        else:
            data = zoom(data, zoom=[1,1.0/zoom_level,1.0/zoom_level,1])
            return data


    def close(self):

        self.hdf5.close()


if __name__ == '__main__':

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'

    # internet_archive_download()
    # convert_pdf_to_image()
    preprocess_image(input_directory = linux_media + 'Pages_Images', output_directory = linux_media + 'Pages_Images_Preprocessed')