import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepzine import DeepZine

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'
    pagedata = '~/Local_Data/preloaded_data.hdf5'
    mridata = '~/Github/mri_GAN/mri_slice.hdf5'

    gan = DeepZine(train=False,
                    test=True,
                    train_data_directory=None,
                    train_hdf5=pagedata,
                    train_overwrite=False,
                    train_preloaded=True,
                    gan_samples_dir='./samples_tiny_latent',
                    gan_log_dir='./log_tiny_latent',
                    test_data_directory='./gridspace_samples',
                    test_model_path='~/Github/dreamTexts/log_tiny_latent/8/model.ckpt',
                    test_model_samples=1000)

    gan.execute()