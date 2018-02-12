import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    from deepzine import DeepZine

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'
    pagedata = '~/Local_Data/preloaded_data.hdf5'
    mridata = '~/Github/mri_GAN/mri_slice.hdf5'

    gan = DeepZine(train=False,
                    test=True,
                    train_data_directory='/mnt/jk489/James/plus_classification/results_with_200_excluded/datasets/FullDataset/raw/Posterior',
                    train_hdf5='./rop_masks.hdf5',
                    train_overwrite=False,
                    train_preloaded=True,
                    train_preprocess_images=True,
                    gan_samples_dir='/mnt/jk489/QTIM_Databank/Test_Outputs/samples_rop_mask',
                    gan_log_dir='./log_rop_masks',
                    test_data_directory='./rop_slerp',
                    test_model_path='./log_rop_masks/8/model.ckpt',
                    test_model_samples=200)

    gan.execute()