
import os
import tables

from shutil import rmtree
from download_data import internet_archive_download, convert_pdf_to_image, preprocess_image, store_to_hdf5, PageData
from util import add_parameter
from model import PGGAN

class DeepZine(object):

    def __init__(self, **kwargs):

        # General Parameters
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'train', True)
        add_parameter(self, kwargs, 'test', True)
        add_parameter(self, kwargs, 'reverse', True)
        add_parameter(self, kwargs, 'encode', True)
        add_parameter(self, kwargs, 'test_encode', True)
        add_parameter(self, kwargs, 'save', True)


        # Model Parameters -- more important for test/reverse, maybe
        add_parameter(self, kwargs, 'channels', 3)
        add_parameter(self, kwargs, 'only_channel', None)
        add_parameter(self, kwargs, 'progressive_depth', 1)

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
        add_parameter(self, kwargs, 'train_classes', False)
        add_parameter(self, kwargs, 'train_channel', None)
        add_parameter(self, kwargs, 'train_classify', None)
        add_parameter(self, kwargs, 'train_categorical_classes', None)

        # Training Classifier Parameters
        # TODO

        # Training GAN Parameters
        add_parameter(self, kwargs, 'gan_samples_dir', './samples')
        add_parameter(self, kwargs, 'gan_log_dir', './log')
        add_parameter(self, kwargs, 'gan_latent_size', 512)
        add_parameter(self, kwargs, 'gan_max_filter', 1024)

        # Test Output Directory
        add_parameter(self, kwargs, 'test_data_directory', None)
        add_parameter(self, kwargs, 'test_model_path', None)
        add_parameter(self, kwargs, 'test_model_samples', 100)
        add_parameter(self, kwargs, 'test_input_latent', 100)

        # Reverse Gan Parameters
        add_parameter(self, kwargs, 'reverse_model_path', None)
        add_parameter(self, kwargs, 'reverse_data_directory', None)
        add_parameter(self, kwargs, 'reverse_batch_size', 8)

        # Encoder Parameters
        add_parameter(self, kwargs, 'encoder_input_model_path', None)
        add_parameter(self, kwargs, 'encoder_batch_size', 8)
        add_parameter(self, kwargs, 'encoder_log_dir', None)
        add_parameter(self, kwargs, 'encoder_samples_dir', None)
        add_parameter(self, kwargs, 'encoder_progressive_depth', 8)

        # Test Encoder Parameters
        add_parameter(self, kwargs, 'test_encode_batch_size', 1)
        add_parameter(self, kwargs, 'test_encode_progressive_depth', 8)
        add_parameter(self, kwargs, 'test_encode_sample_dir', None)

        # Save Parameters
        add_parameter(self, kwargs, 'save_input_directory', None)
        add_parameter(self, kwargs, 'save_output_directory', None)
        add_parameter(self, kwargs, 'save_progressive_depth', 8)

        return

    def execute(self):

        if self.train:

            # Data preparation.
            self.training_storage = self.download_data(data_directory=self.train_data_directory, download_pdf=self.train_download_pdf, internetarchive_collection=self.train_internetarchive_collection, convert_pdf=self.train_convert_pdf, preprocess_images=self.train_preprocess_images, preprocess_shape=self.train_preprocess_shape, hdf5=self.train_hdf5, overwrite=self.train_overwrite, preloaded=self.train_preloaded)

            if True:
            # try:
                self.train_gan()
            # except:
                self.training_storage.close()

            self.training_storage.close()

        if self.test:

            if True:
                self.test_gan()

        if self.reverse:

            if True:
                self.reverse_gan()

        if self.encode:

            if True:
                self.train_encoder()

        if self.test_encode:

            if True:
                self.test_encoder()

        if self.save:

            if True:
                self.save_model()

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

        # Preprocess Images. TODO (different preprocessing methods)
        # if preprocess_images:
        #     preprocessed_directory = os.path.join('.', 'rop_images')
        #     if not os.path.exists(preprocessed_directory):
        #         os.mkdir(preprocessed_directory)
        #     preprocess_image(data_directory, preprocessed_directory)
        # else:
        #     preprocessed_directory = preprocessed_directory

        # Convert to HDF5
        if not os.path.exists(hdf5) or overwrite:
            output_hdf5 = store_preloaded_hdf5_file(preprocessed_directory, hdf5, preprocess_shape)
        else:
            output_hdf5 = tables.open_file(hdf5, "r")

        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        return PageData(hdf5=output_hdf5, shape=preprocess_shape, collection=internetarchive_collection, preloaded=preloaded, preprocessed=preprocess_images, channel=self.train_channel)

    def train_gan(self):

        # Create necessary directories
        for work_dir in [self.gan_samples_dir, self.gan_log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Inherited this from other code, think of a programmatic way to do it.
        training_depths = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        read_depths = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9]
        # training_depths = [5,5,6,6,7,7,8,8,9,9]
        # read_depths = [4,5,5,6,6,7,7,8,8,9]
        # training_depths = [9,9]
        # read_depths = [8,9]
        # training_depths = [9]
        # read_depths = [9]
        # training_depths = [4,4,5,5,6,6,7,7,8,8,9,9]
        # read_depths = [3,4,4,5,5,6,6,7,7,8,8,9]

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
                            transition=transition,
                            channels=self.channels,
                            classes=self.train_classes,
                            categorical_classes=self.train_categorical_classes,
                            classify=self.train_classify)

            pggan.build_model()

            if self.train_classify:
                pggan.train_cgan()
            else:
                pggan.train()

    def test_gan(self, input_latent=None):

        if not os.path.exists(self.test_data_directory):
            os.makedirs(self.test_data_directory)

        if self.test_input_latent is None:
            pggan = PGGAN(input_model_path=self.test_model_path,
                            progressive_depth=self.progressive_depth,
                            testing=True,
                            channel=self.channels)

            pggan.build_model()
            pggan.test_model(self.test_data_directory, self.test_model_samples)      

        else:
            for i in xrange(1,9):
                pggan = PGGAN(input_model_path=os.path.join(self.test_model_path, str(i), 'model.ckpt'),
                                progressive_depth=i,
                                testing=True,
                                channels=self.channels)

                pggan.build_model()
                pggan.test_model(self.test_data_directory, self.test_model_samples, input_latent=self.test_input_latent)      


    def reverse_gan(self, progressive = True):

        output_hdf5 = tables.open_file(self.train_hdf5, "r")
        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        data_loader = PageData(hdf5=output_hdf5, preloaded=True)

        if not os.path.exists(self.reverse_data_directory):
            os.makedirs(self.reverse_data_directory)

        if not progressive:
            pggan = PGGAN(training_data=data_loader,
                            input_model_path=self.reverse_model_path,
                            progressive_depth=self.progressive_depth,
                            channels=self.channels)

            pggan.build_model()
            pggan.fit_latent_vector(self.reverse_data_directory)      
        else:

            initialized_latent = None
            for i in xrange(1,9):

                output_directory = os.path.join(self.reverse_data_directory, str(i))
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                pggan = PGGAN(training_data=data_loader,
                            input_model_path=os.path.join(self.reverse_model_path, str(i), 'model.ckpt'),
                            progressive_depth=i,
                            channels=self.channels,
                            batch_size=self.reverse_batch_size)

                pggan.build_model()
                initialized_latent = pggan.fit_latent_vector(output_directory, initialized_latent=initialized_latent)     

    def train_encoder(self, progressive=False):

        for work_dir in [self.encoder_samples_dir, self.encoder_log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Inherited this from other code, think of a programmatic way to do it.
        # training_depths = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        # read_depths = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9]
        # read_generator_depths = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
        training_depths = [7]
        read_depths = [7]
        read_generator_depths = [7]
        # training_depths = [9]
        # read_depths = [9]
        # training_depths = [4,4,5,5,6,6,7,7,8,8,9,9]
        # read_depths = [3,4,4,5,5,6,6,7,7,8,8,9]

        for i in range(len(training_depths)):

            if (i % 2 == 0):
                transition = False
            else:
                transition = True

            output_encoder_model_path = os.path.join(self.encoder_log_dir, str(training_depths[i]), 'model.ckpt')
            if not os.path.exists(os.path.dirname(output_encoder_model_path)):
                os.makedirs(os.path.dirname(output_encoder_model_path))

            input_encoder_model_path = os.path.join(self.encoder_log_dir, str(read_depths[i]), 'model.ckpt')

            input_model_path = os.path.join(self.encoder_input_model_path, str(read_generator_depths[i]), 'model.ckpt')

            sample_path = os.path.join(self.encoder_samples_dir, 'sample_' + str(training_depths[i]) + '_' + str(transition))
            if not os.path.exists(sample_path):
                os.mkdir(sample_path)

            pggan = PGGAN(training_data=None,
                            input_model_path=input_model_path,
                            input_encoder_model_path=input_encoder_model_path,
                            output_encoder_model_path=output_encoder_model_path,
                            samples_dir=sample_path, 
                            encoder_log_dir=self.encoder_log_dir,
                            progressive_depth=training_depths[i],
                            channels=self.channels,
                            transition=transition,
                            encoding=True,
                            progressive=False)

            pggan.build_model()
            pggan.train_encoder()

        return

    def test_encoder(self):

        output_hdf5 = tables.open_file(self.train_hdf5, "r")
        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        data_loader = PageData(hdf5=output_hdf5, preloaded=True)

        if not os.path.exists(self.test_encode_sample_dir):
            os.makedirs(self.test_encode_sample_dir)

        pggan = PGGAN(training_data=data_loader,
                        input_encoder_model_path=os.path.join(self.encoder_log_dir, str(self.test_encode_progressive_depth), 'model.ckpt'),
                        progressive_depth=self.test_encode_progressive_depth,
                        input_model_path=os.path.join(self.encoder_input_model_path, str(self.test_encode_progressive_depth), 'model.ckpt'),
                        channels=self.channels,
                        samples_dir=self.test_encode_sample_dir,
                        batch_size=self.test_encode_batch_size,
                        testing=True,
                        encoding=True,
                        only_channel=self.only_channel)

        pggan.build_model()
        pggan.test_encoder()      

    def save_model(self):

        input_model_path = os.path.join(self.save_input_directory, str(self.save_progressive_depth), 'model.ckpt')
        output_model_path = os.path.join(self.save_output_directory, str(self.save_output_directory), str(self.save_progressive_depth))

        if os.path.exists(output_model_path):
            rmtree(output_model_path)

        if not os.path.exists(self.save_output_directory):
            os.makedirs(self.save_output_directory)

        pggan = PGGAN(input_model_path=input_model_path,
                        progressive_depth=self.save_progressive_depth,
                        channels=self.channels,
                        save_model_path=output_model_path,
                        batch_size=1)

        pggan.build_model()
        pggan.save_model()

if __name__ == '__main__':

    pass