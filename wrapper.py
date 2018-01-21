import os

from deepzine import DeepZine

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    gan = DeepZine(train_data_directory='E:/Pages_Images_Preprocessed',
                    train_hdf5='E:/Pages_HDF5/pages.hdf5')

    gan.execute()