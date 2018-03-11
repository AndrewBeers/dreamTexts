import os
import numpy as np

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepzine import DeepZine

    windows_media = 'E:/'
    pagedata = '~/Local_Data/preloaded_data.hdf5'
    mridata = '~/Github/mri_GAN/mri_slice.hdf5'

    input_latent = np.random.normal(size=[8,128])



    gan = DeepZine(channels=3,
                    progressive_depth=8,
                    train=False,
                    test=False,
                    reverse=False,
                    encode=False,
                    test_encode=False,
                    save=True,
                    train_data_directory='/raw/Posterior',
                    train_hdf5='./rop_multiclass_with_cats.hdf5',
                    train_overwrite=False,
                    train_preloaded=True,
                    train_preprocess_images=False,
                    train_classes=True,
                    train_channel=None,
                    train_classify=None,
                    train_categorical_classes=None,
                    gan_samples_dir='/Test_Outputs/samples_rop_seg_and_rgb_plus_only',
                    gan_log_dir='./log_rop_seg_and_rgb_plus_only',
                    test_data_directory='./rop_same_latents',
                    test_model_path='./log_rop_masks/',
                    test_model_samples=200,
                    test_input_latent=input_latent,
                    reverse_model_path='./log_rop_masks/',
                    reverse_data_directory='rop_reversed',
                    reverse_batch_size=4,
                    encoder_input_model_path='./log_rop_masks',
                    encoder_log_dir='./log_rop_masks_encoder_test',
                    encoder_samples_dir='./log_rop_masks_encoder_samples',
                    encoder_batch_size=8,
                    encoder_progressive_depth=8,
                    save_input_directory='/home/local/PARTNERS/azb22/Github/dreamTexts/log_smaller/',
                    save_output_directory='/home/local/PARTNERS/azb22/Github/dreamTexts/saved_pages/',
                    save_progressive_depth=9)



    '''

    gan = DeepZine(channels=4,
                    progressive_depth=8,
                    train=True,
                    test=False,
                    reverse=False,
                    encode=False,
                    train_data_directory='/raw/Posterior',
                    train_hdf5='./BRATS_with_seg_pp.hdf5',
                    train_overwrite=False,
                    train_preloaded=True,
                    train_preprocess_images=False,
                    train_classes=None,
                    train_channel=None,
                    train_classify=None,
                    train_categorical_classes=None,
                    gan_samples_dir='/Test_Outputs/samples_mri_slice_no_seg',
                    gan_log_dir='./log_samples_mri_slice_no_seg',
                    test_data_directory='./rop_same_latents',
                    test_model_path='./log_rop_masks/',
                    test_model_samples=200,
                    test_input_latent=input_latent,
                    reverse_model_path='./log_rop_masks/',
                    reverse_data_directory='rop_reversed',
                    reverse_batch_size=4,
                    encoder_input_model_path='./log_rop_masks',
                    encoder_log_dir='./log_rop_masks_encoder_test',
                    encoder_samples_dir='./log_rop_masks_encoder_samples',
                    encoder_batch_size=8,
                    encoder_progressive_depth=8)

    '''

    # gan = DeepZine(channels=5,
    #                 only_channel=None,
    #                 progressive_depth=7,
    #                 train=False,
    #                 test=False,
    #                 reverse=False,
    #                 encode=True,
    #                 test_encode=False,
    #                 train_data_directory='/raw/Posterior',
    #                 train_hdf5='./rop_1st_100.hdf5',
    #                 train_overwrite=False,
    #                 train_preloaded=True,
    #                 train_preprocess_images=True,
    #                 gan_samples_dir='/Test_Outputs/samples_rop_mask',
    #                 gan_log_dir='./log_rop_masks',
    #                 test_data_directory='./rop_same_latents',
    #                 test_model_path='./log_rop_masks/',
    #                 test_model_samples=200,
    #                 test_input_latent=input_latent,
    #                 reverse_model_path='./log_rop_masks/',
    #                 reverse_data_directory='rop_reversed',
    #                 reverse_batch_size=4,
    #                 encoder_input_model_path='./log_mri_slice_seg_pp',
    #                 encoder_log_dir='./log_mri_slice_seg_pp_encoder',
    #                 encoder_samples_dir='./log_mri_slice_seg_pp_encoder_samples',
    #                 encoder_batch_size=8,
    #                 encoder_progressive_depth=7,
    #                 test_encode_batch_size=1,
    #                 test_encode_progressive_depth=8,
    #                 test_encode_sample_dir='./log_reverse_real_rop_latent_vessels')
    

    gan.execute()