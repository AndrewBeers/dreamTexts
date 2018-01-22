import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepzine import DeepZine

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'

    gan = DeepZine(train_data_directory=linux_media + 'Pages_Images_Preprocessed',
                    train_hdf5=linux_media + 'Pages_HDF5/pages.hdf5',
                    train_overwrite=False)

    gan.execute()