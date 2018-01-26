import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    from deepzine import DeepZine

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'

    gan = DeepZine(train_data_directory=None,
                    train_hdf5='preloaded_pages_alt.hdf5',
                    train_overwrite=False,
                    train_preloaded=True,
                    gan_samples_dir='./samples_smaller',
                    gan_log_dir='./log_smaller')

    gan.execute()