
import os
import tables

from download_data import internet_archive_download, convert_pdf_to_image, preprocess_image, store_to_hdf5
from util import add_parameter

class DeepZine(object):

    def __init__(self, **kwargs):

        # General Parameters
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'train', True)
        add_parameter(self, kwargs, 'test', True)

        # Train Data Parameters
        add_parameter(self, kwargs, 'train_data_directory', None)
        add_parameter(self, kwargs, 'train_download_pdf', False)
        add_parameter(self, kwargs, 'train_internetarchive_collection', None)
        add_parameter(self, kwargs, 'train_convert_pdf', False)
        add_parameter(self, kwargs, 'train_preprocess_images', False)
        add_parameter(self, kwargs, 'train_preprocess_shape', (64, 64))
        add_parameter(self, kwargs, 'train_hdf5', None)
        add_parameter(self, kwargs, 'train_overwrite', False)

        # Training Classified Parameters
        # TODO

        # Training GAN Parameters


        return

    def execute(self):

        if self.train:

            # Data preparation.
            self.training_storage = self.download_data(data_directory=self.train_data_directory, download_pdf=self.train_download_pdf, internetarchive_collection=self.train_internetarchive_collection, convert_pdf=self.train_convert_pdf, preprocess_images=self.train_preprocess_images, preprocess_shape=self.train_preprocess_shape, hdf5=self.train_hdf5, overwrite=self.train_overwrite)

            # model = self.create_model('upsampling_gan')

            self.training_storage.close()

        return

    def download_data(self, data_directory=None, download_pdf=False, internetarchive_collection=None, convert_pdf=False, preprocess_images=False, preprocess_shape=(64, 64), hdf5=None, overwrite=False):

        # The goal here is to return an HDF5 we can stream from.
        if hdf5 is not None and data_directory is None:
            if os.path.exists(hdf5):
                return load_hdf5(hdf5)
            else:
                raise ValueError('Input HDF5 file not found.')

        # Create a working data_directory if necessary.
        if not os.path.exists(data_directory) and not download_pdf:
            raise ValueError('Data directory not found.')
        elif not os.path.exists(data_directory):
            os.mkdir(data_directory)

        # Download data
        if download_pdf:
            internet_archive_download(data_directory, internetarchive_collection)

        # Convert PDFs
        if convert_pdf:
            converted_directory = os.path.join(data_directory, 'converted_images')
            if not os.path.exists(converted_directory):
                os.mkdir(converted_directory)
            convert_pdf_to_image(data_directory, converted_directory)
        else:
            converted_directory = data_directory

        # Preprocess Images. TODO (different preprocessing methods)
        if preprocess_images:
            preprocessed_directory = os.path.join(data_directory, 'converted_images')
            if not os.path.exists(converted_directory):
                os.mkdir(converted_directory)
            convert_pdf_to_image(data_directory, converted_directory)
        else:
            preprocessed_directory = converted_directory

        # Convert to HDF5
        if not os.path.exists(hdf5) or overwrite:
            output_hdf5 = store_to_hdf5(preprocessed_directory, hdf5, preprocess_shape)
            return output_hdf5
        else:
            return tables.open_file(hdf5, "r")

    def create_model(self):

        return


if __name__ == '__main__':

    pass