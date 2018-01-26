
import os
import tables

from download_data import internet_archive_download, convert_pdf_to_image, preprocess_image, store_to_hdf5, PageData
from util import add_parameter
from model import PGGAN

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
        add_parameter(self, kwargs, 'train_preloaded', False)

        # Training Classifier Parameters
        # TODO

        # Training GAN Parameters
        add_parameter(self, kwargs, 'gan_samples_dir', './samples')
        add_parameter(self, kwargs, 'gan_log_dir', './log')
        add_parameter(self, kwargs, 'gan_latent_size', 512)
        add_parameter(self, kwargs, 'gan_max_filter', 1024)

        return

    def execute(self):

        if self.train:

            # Data preparation.
            self.training_storage = self.download_data(data_directory=self.train_data_directory, download_pdf=self.train_download_pdf, internetarchive_collection=self.train_internetarchive_collection, convert_pdf=self.train_convert_pdf, preprocess_images=self.train_preprocess_images, preprocess_shape=self.train_preprocess_shape, hdf5=self.train_hdf5, overwrite=self.train_overwrite, preloaded=self.train_preloaded)

            if True:
            # try:
                self.train_gan()
            # except:
                # self.training_storage.close()


            self.training_storage.close()

        return

    def download_data(self, data_directory=None, download_pdf=False, internetarchive_collection=None, convert_pdf=False, preprocess_images=False, preprocess_shape=(64, 64), hdf5=None, overwrite=False, preloaded=False):

        # Temporary Commenting

        # # The goal here is to return an HDF5 we can stream from.
        # if hdf5 is not None and data_directory is None:
        #     if os.path.exists(hdf5):
        #         output_hdf5 = hdf
        #     else:
        #         raise ValueError('Input HDF5 file not found.')

        # # Create a working data_directory if necessary.
        # if not os.path.exists(data_directory) and not download_pdf:
        #     raise ValueError('Data directory not found.')
        # elif not os.path.exists(data_directory):
        #     os.mkdir(data_directory)

        # # Download data
        # if download_pdf:
        #     internet_archive_download(data_directory, internetarchive_collection)

        # # Convert PDFs
        # if convert_pdf:
        #     converted_directory = os.path.join(data_directory, 'converted_images')
        #     if not os.path.exists(converted_directory):
        #         os.mkdir(converted_directory)
        #     convert_pdf_to_image(data_directory, converted_directory)
        # else:
        #     converted_directory = data_directory

        # # Preprocess Images. TODO (different preprocessing methods)
        # if preprocess_images:
        #     preprocessed_directory = os.path.join(data_directory, 'converted_images')
        #     if not os.path.exists(converted_directory):
        #         os.mkdir(converted_directory)
        #     convert_pdf_to_image(data_directory, converted_directory)
        # else:
        #     preprocessed_directory = converted_directory

        # Convert to HDF5
        if not os.path.exists(hdf5) or overwrite:
            output_hdf5 = store_to_hdf5(preprocessed_directory, hdf5, preprocess_shape)
        else:
            output_hdf5 = tables.open_file(hdf5, "r")

        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        return PageData(hdf5=output_hdf5, shape=preprocess_shape, collection=internetarchive_collection, preloaded=preloaded)

    def train_gan(self):

        # Create necessary directories
        for work_dir in [self.gan_samples_dir, self.gan_log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Inherited this from other code, think of a programmatic way to do it.
        training_depths = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        read_depths = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9]
        # training_depths = [9]
        # read_depths = [9]

        for i in range(len(training_depths)):

            if (i % 2 == 0):
                transition = False
            else:
                transition = True

            output_model_path = os.path.join(self.gan_log_dir, str(training_depths[i]), 'model.ckpt')
            if not os.path.exists(os.path.dirname(output_model_path)):
                os.mkdir(os.path.dirname(output_model_path))

            input_model_path = os.path.join(self.gan_log_dir, str(read_depths[i]), 'model.ckpt')

            sample_path = os.path.join(self.gan_samples_dir, 'sample_' + str(training_depths[i]) + '_' + str(transition))
            if not os.path.exists(sample_path):
                os.mkdir(sample_path)

            pggan = PGGAN(training_data = self.training_storage,
                            input_model_path=input_model_path, 
                            output_model_path=output_model_path,
                            samples_dir=sample_path, 
                            log_dir=self.gan_log_dir,
                            progressive_depth=training_depths[i],
                            transition=transition)

            pggan.build_model()
            pggan.train()

if __name__ == '__main__':

    pass