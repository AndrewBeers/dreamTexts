import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import numpy as np
import math
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tables
import random

from scipy.misc import imresize
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from scipy import spatial

from qtim_tools.qtim_utilities.file_util import grab_files_recursive

from deepzine import DeepZine
from util import add_parameter, save_images, save_image
from model import PGGAN
from download_data import internet_archive_download, convert_pdf_to_image, preprocess_image, store_to_hdf5, PageData

depth_dictionary = {1:4, 2:8}

def sample_gan(model_directory, output_directory=None, output_num=1, depth=8, channels=4, batch_size=8, input_csv=None, sample_latent=None, input_npy=None, random_variation=.1, random_variation_num=None, output_csv=None, output_npy=None, latent_size=128):

    model_file = os.path.join(model_directory, str(depth), 'model.ckpt')

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    pggan = PGGAN(input_model_path=model_file,
                    progressive_depth=depth,
                    testing=True,
                    batch_size=batch_size,
                    latent_size=latent_size,
                    channels=channels)

    batch_size = pggan.batch_size

    pggan.build_model()   

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # Personally have no idea what is being logged in this thing --andrew
        sess.run(init)

        pggan.saver.restore(sess, model_file)

        # Load inputs, whatever format they may take.
        if sample_latent is None:
            if input_csv is not None:
                sample_latent = np.loadtxt(input_csv, dtype=float)
            elif input_npy is not None:
                sample_latent = np.load(input_npy)
            else:
                sample_latent = np.random.normal(size=[output_num, pggan.latent_size])

        if random_variation_num is not None:
            variation_sample_latent = np.zeros((output_num*random_variation_num, pggan.latent_size), dtype=float)
            for latent_idx in xrange(output_num):
                for variation_idx in xrange(random_variation_num):
                    variation_sample_latent[latent_idx + variation_idx] = sample_latent[latent_idx] + np.random.normal(0, random_variation, pggan.latent_size)
            sample_latent = variation_sample_latent
            output_num *= random_variation_num

        # Save to .npy file if you want.
        if output_npy is not None:
            np.save(output_npy, sample_latent)

        # Save latents to csv if you want.
        if output_csv is not None:
            with open(output_csv, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for row in sample_latent:
                    writer.writerow(row)

        output_images = np.zeros((output_num, 2**(depth+1), 2**(depth+1), channels))

        for idx in np.arange(output_num, step=batch_size, dtype=int):
            fake_image = sess.run(pggan.fake_images, feed_dict={pggan.latent: sample_latent[idx:idx+batch_size]})
            fake_image = np.clip(fake_image, -1, 1)

            output_images[idx:idx+batch_size] = fake_image
            print 'Generating image...', idx

            for image_id, image in enumerate(fake_image):
                save_image(image[np.newaxis,...], '{}/{:02d}_generated_image.png'.format(output_directory, idx+image_id))

    tf.reset_default_graph()

    return output_images, sample_latent

        # else:
            # return


def encode_images(encoder_model_directory, gan_model_directory, images_directory='/mnt//James/plus_classification/results_with_200_excluded/datasets/First100/raw', batch_size=10, mask_directory='/mnt//James/plus_classification/results_with_200_excluded/datasets/First100/output', output_directory=None, output_csv='test.csv', output_npy=None, input_hdf5=None, channels=4, only_channel=-1, depth=7, input_num=1000):

    np.set_printoptions(suppress=True)

    # If you wanted to use hdf5... Currently unimplemented.
    if input_hdf5 is not None:
        output_hdf5 = tables.open_file(self.train_hdf5, "r")
        data_loader = PageData(hdf5=input_hdf5, preloaded=True)

    # Create output directory.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Create PGGAN
    pggan = PGGAN(input_encoder_model_path=os.path.join(encoder_model_directory, str(depth), 'model.ckpt'),
                    progressive_depth=depth,
                    input_model_path=os.path.join(gan_model_directory, str(depth), 'model.ckpt'),
                    channels=channels,
                    samples_dir=output_directory,
                    batch_size=batch_size,
                    testing=True,
                    encoding=True,
                    only_channel=only_channel)
    pggan.build_model()
    batch_size = pggan.batch_size

    # Load in all the images.
    if False:
        images = grab_files_recursive(images_directory)

        if input_num is not None:
            random.shuffle(images)
            images = images[:input_num]

        images_array = np.zeros((len(images), 512, 512, channels), dtype=float)
        imagenames_array = np.zeros((len(images)), dtype=object)
        imageclasses_array = np.zeros((len(images)), dtype=object)
        for image_idx, image in enumerate(images):
            
            imagenames_array[image_idx] = image

            img = np.asarray(Image.open(image))[...,0:3]
            if mask_directory is not None:
                # if True:
                try:
                    img_mask = np.asarray(Image.open(os.path.join(mask_directory, os.path.basename(image)[0:-3] + 'png')))
                except:
                    continue

            img = imresize(img, (512, 512))
            img_mask = imresize(img_mask, (512, 512))[..., np.newaxis]
            data = np.concatenate((img, img_mask), axis=2)
            images_array[image_idx] = data

            # if '-os' in os.path.basename(image) or '-OS' in os.path.basename(image):
            #     imageclasses_array[image_idx] = 0
            # elif '-od' in os.path.basename(image) or '-OD' in os.path.basename(image):
            #     imageclasses_array[image_idx] = 1
            # else:
            #     imageclasses_array[image_idx] = 2
            
            # if os.path.basename(os.path.dirname(image)) == 'normal':
            #     imageclasses_array[image_idx] = 0
            # elif os.path.basename(os.path.dirname(image)) == 'pre-plus':
            #     imageclasses_array[image_idx] = 1
            # else:
            #     imageclasses_array[image_idx] = 2

            print image
        input_num = len(images)
    else:

        if output_npy is not None and False:
            images_array = np.load(output_npy[0])
            imageclasses_array = np.load(output_npy[1])
            imagenames_array = np.load(output_npy[2])
        else:

            images = grab_files_recursive(images_directory, 'slice_0*')
            random.shuffle(images)
            images = images[:input_num]

            images_array = np.zeros((input_num, 256, 256, 5), dtype=float)
            imagenames_array = np.zeros((len(images)), dtype=object)
            imageclasses_array = np.zeros((len(images), 4), dtype=object)
            for image_idx, image in enumerate(images):

                seg_image = np.asarray(Image.open(image))[...,np.newaxis]
                try:
                    for i in [4,3,2,1]:
                        new_filename = image.replace('slice_0', 'slice_' + str(i))
                        new_array = np.asarray(Image.open(new_filename))[...,np.newaxis]
                        seg_image = np.concatenate((new_array, seg_image), axis=2)
                except:
                    print 'error!'
                    imagenames_array[image_idx] = 'empty'
                    continue

                imagenames_array[image_idx] = image
                images_array[image_idx] = seg_image

                split_image = str.split(os.path.basename(image), '_')
                classes = [float(split_image[i]) for i in [2,3,4,5]]
                imageclasses_array[image_idx] = classes

                print image

            if output_npy is not None:
                np.save(output_npy[0], images_array)
                np.save(output_npy[1], imageclasses_array)
                np.save(output_npy[2], imagenames_array)

    images_array = images_array / 127.5 - 1

    # Pre-initialize output array.
    if output_csv is not None or output_npy is not None:
        outputs = np.zeros((input_num, pggan.latent_size), dtype=object)

    # Tensorflow options.
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if output_csv is not None and False:
        outputs = np.loadtxt(output_csv, delimiter=',', dtype=object)[:,1:].astype(float)
    else:
        with tf.Session(config=config) as sess:

            sess.run(init)

            pggan.saver.restore(sess, pggan.input_model_path)
            pggan.encoder_saver.restore(sess, pggan.input_encoder_model_path)

            for idx in np.arange(input_num, step=batch_size, dtype=int):

                target_image = images_array[idx:idx + batch_size]

                if np.sum(target_image) == 0:
                    continue

                if pggan.only_channel is None:
                    generated_latent = sess.run(pggan.decoded_latent, feed_dict={pggan.images: target_image})
                else:
                    generated_latent = sess.run(pggan.decoded_latent, feed_dict={pggan.seg_images: target_image[...,pggan.only_channel][...,np.newaxis]})

                outputs[idx:idx+batch_size] = generated_latent
                print generated_latent[0]

                generated_images = sess.run(pggan.fake_images, feed_dict={pggan.latent: generated_latent})

                image_outputs = np.concatenate((target_image, generated_images), axis=0)
                image_outputs = np.clip(image_outputs, -1, 1)

                if batch_size == 1:
                    generated_images = np.clip(generated_images, -1, 1)
                    output_filename = imagenames_array[idx]
                    output_filename = os.path.join(output_directory, os.path.basename(output_filename))
                    save_image(generated_images, output_filename)
                else:
                    output_filename = '{}/{:02d}_decoded.png'.format(output_directory, idx)
                    save_images(image_outputs, [2, pggan.batch_size], output_filename)

    # Save latents to csv if you want.
    if output_csv is not None:
        with open(output_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for row_idx, row in enumerate(outputs):
                writer.writerow([imagenames_array[row_idx]] + row.tolist())

    # print(pca.explained_variance_)

    # for perplexity in [5, 10, 15, 20, 25, 30, 35]:
    for perplexity in [50, 60, 80, 100, 35]:

        tsne = TSNE(n_components=2, perplexity=perplexity)
        tsne.fit(outputs)

        projected = tsne.fit_transform(outputs)

        # fig = plt.figure()
        # ax = Axes3D(fig)

        # ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                    # c=imageclasses_array[:,-1], edgecolor='none', alpha=0.5,
                    # cmap=plt.cm.get_cmap('spectral', 10))
        # ax.set_xlabel('component 1')
        # ax.set_ylabel('component 2')
        # ax.set_zlabel('component 2')
        # # fig.colorbar()
        # plt.show()
    # pca = PCA(n_components=50)
    # pca.fit(outputs)
    # print(pca.explained_variance_)

    # projected = pca.fit_transform(outputs)

        plt.scatter(projected[:, 0], projected[:, 1], c=imageclasses_array[:,-1], edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('RdBu_r', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        # plt.show()
        plt.colorbar()
        plt.show()

    return

def visualize_latents(input_csv=None, method='pca'):

    if input_csv is not None:
        samples = np.zeros

    if method == 'pca':
        return

def tumor_latent(input_npy='generated_mri.npy', input_csv='mri_latents.npy'):

    outputs = np.loadtxt(input_csv, delimiter=',', dtype=object)
    filenames = outputs[:, 0]
    latents = outputs[:,1:].astype(float)
    classes = np.zeros((filenames.shape), dtype=float)

    for idx, filename in enumerate(filenames):
        print filename
        # classes[idx] = int(str.split(os.path.basename(filename), '_')[-2])

        edema_value = float(str.split(os.path.basename(filename), '_')[-5])
        enhancing_value = float(str.split(os.path.basename(filename), '_')[-4])
        print edema_value, enhancing_value
        classes[idx] = enhancing_value / float(edema_value + enhancing_value)
        print classes[idx]
        # if '_Pre-Plus' in os.path.basename(filename):
            # classes[idx] = 1
        # elif '_Plus' in os.path.basename(filename):
            # classes[idx] = 2

    # model = PCA(n_components=2)
    # model.fit(latents)
    # print model.explained_variance_

    # if input_model is not None:
    #     model = joblib.load(input_model) 
    # else:
    # model = TSNE(n_components=2, perplexity=50, verbose=True)
    # model.fit(latents)

    # projected = model.fit_transform(latents)
    projected = np.load('projected_mri.npy')
    # np.save('projected_mri.npy', projected)

    plt.scatter(projected[:, 0], projected[:, 1], c=classes, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('viridis'))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    # plt.show()
    plt.colorbar()
    plt.show()

    # if output_model is not None:
        # joblib.dump(model, output_model)

def retina_latent(input_csv='retinal_latents.csv', input_model=None, output_model=None):

    outputs = np.loadtxt(input_csv, delimiter=',', dtype=object)
    filenames = outputs[:, 0]
    latents = outputs[:,1:].astype(float)
    classes = np.zeros((filenames.shape), dtype=int)

    for idx, filename in enumerate(filenames):
        if '_os_' in os.path.basename(filename):
            classes[idx] = 1
        # if '_Pre-Plus' in os.path.basename(filename):
            # classes[idx] = 1
        # elif '_Plus' in os.path.basename(filename):
            # classes[idx] = 2

    # model = PCA(n_components=2)
    # model.fit(latents)
    # print model.explained_variance_

    # if input_model is not None:
    #     model = joblib.load(input_model) 
    # else:
    model = TSNE(n_components=2, perplexity=50, verbose=True, )
    model.fit(latents)

    projected = model.fit_transform(latents)
    np.save('projected.npy', projected)

    plt.scatter(projected[:, 0], projected[:, 1], c=classes, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('viridis', len(np.unique(classes))))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    # plt.show()
    plt.colorbar()
    plt.show()

    if output_model is not None:
        joblib.dump(model, output_model)

def find_projected(input_csv='retinal_latents.csv', input_projected='projected.npy'):

    outputs = np.loadtxt(input_csv, delimiter=',', dtype=object)
    print outputs.shape
    filenames = outputs[:, 0]
    latents = outputs[:,1:].astype(float)
    classes = np.zeros((filenames.shape), dtype=int)

    projected_data = np.load(input_projected)
    print projected_data.shape

    tree = spatial.KDTree(projected_data)
    output = tree.query([(-25, 5)])
    file = filenames[output[1]][0]

    file = file.replace('slice_0', 'slice_2')
    print file
    image = Image.open(file)
    image.show()

    for row_idx, row in enumerate(projected_data):
        # print int(row[0]), int(row[1])
        if int(row[0]) == -5 and int(row[1]) == -30:
            print filenames[row_idx]



def get_discriminator_scores(discriminator_model, input_images=None, input_hdf5=None, batch_size=1, output_csv=None, image_num=None):

    np.set_printoptions(suppress=True)

    # If you wanted to use hdf5... Currently unimplemented.
    if input_hdf5 is not None:
        output_hdf5 = tables.open_file(input_hdf5, "r") 
        data_loader = PageData(hdf5=output_hdf5, preloaded=True, preprocessed=False, classes=None)

    if input_images is not None:
        image_num = input_images.shape[0]

    model_file = os.path.join(discriminator_model, str(7), 'model.ckpt')

    pggan = PGGAN(input_model_path=discriminator_model,
                    progressive_depth=7,
                    testing=True,
                    batch_size=batch_size,
                    latent_size=128)

    batch_size = pggan.batch_size

    pggan.build_model()   

    scores_output = np.zeros((image_num))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        sess.run(init)

        pggan.saver.restore(sess, model_file)

        for idx in np.arange(image_num, step=batch_size, dtype=int):
            if input_images is not None:
                discrim_score = sess.run(pggan.D_pro_logits, feed_dict={pggan.images: input_images[idx:idx+batch_size,...,0:3]})
            else:
                realbatch_array, realclasses_array = data_loader.get_next_batch(batch_num=idx/batch_size, zoom_level=7, batch_size=batch_size)
                discrim_score = sess.run(pggan.D_pro_logits, feed_dict={pggan.images: realbatch_array})
            print discrim_score
            scores_output[idx:idx+batch_size] = discrim_score[0][0]
        
    # Save latents to csv if you want.
    if output_csv is not None:
        with open(output_csv, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in enumerate(scores_output):
                writer.writerow(row)

    tf.reset_default_graph()

    return

if __name__ == '__main__':

    # GANS
    # retina_rgb_only = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/RGB_Only' # Latent Size is 2 for some reason.
    # retina_rgb_and_seg = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/Segmentations_and_RGB'
    # retina_seg_only = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/Segmentations_Only'
    # mri_slice_no_seg = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/MRI_Slice_no_seg'
    # mri_slice_seg = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/MRI_Slice_with_seg'

    retina_rgb_only = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/RGB_Only' # Latent Size is 2 for some reason.
    retina_rgb_and_seg = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/Segmentations_and_RGB'
    retina_seg_only = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/Segmentations_Only'
    mri_slice_no_seg = '/mnt//QTIM_Experiments/MICCAI_GAN_EXPERIMENTS/Models/MRI_Slice_no_seg'
    mri_slice_seg = '/home/abeers/Projects/PGGAN/MRI_Images/log_mri_slice_seg_pp'

    # NEW MODELS
    retina_rgb_and_seg = '/home/abeers/Projects/PGGAN/Retinal_Images/log_rop_masks'
    retina_rgb = '/home/abeers/Projects/PGGAN/Retinal_Images/log_rop'
    encoding_retina_rgb_and_seg = '/home/abeers/Projects/PGGAN/Retinal_Images/log_rop_masks_encoder_vessels'
    mri_slice_no_seg = '/home/abeers/Projects/PGGAN/MRI_Images/log_samples_mri_slice_no_seg'
    encoder_slice_with_seg = '/home/abeers/Projects/PGGAN/MRI_Images/log_mri_slice_seg_pp_encoder'
    
    # mri_sample_no_seg = sample_gan(model_directory=mri_slice_no_seg, output_num=300, image_size=256)
    # get_discriminator_scores(discriminator_model=mri_slice_no_seg, input_images=mri_sample_no_seg, output_csv='mri_sample_no_seg.csv')
    # mri_sample_seg = sample_gan(model_directory=mri_slice_seg, output_num=240, image_size=256)
    # get_discriminator_scores(discriminator_model=mri_slice_no_seg, input_images=mri_sample_seg, output_csv='mri_sample_seg.csv')
    # get_discriminator_scores(discriminator_model=mri_slice_no_seg, input_hdf5='/home/local/PARTNERS/azb22/Github/mri_GAN/mri_slice.hdf5', output_csv='real_sample_no_seg.csv', image_num=240)
    
    # Generate MRI Slices
    # mri_slices, mri_latents = sample_gan(model_directory=mri_slice_seg, output_directory='/home/abeers/Projects/PGGAN/MRI_Images/random_latent_images', output_num=1000, random_variation_num=None, batch_size=10, depth=7, channels=5, output_csv='random_mri_latents.csv')
    # np.save('generated_mri.npy', mri_slices)
    # np.save('mri_latents.npy', mri_latents)

    # tumor_latent()

    # Generate MRI Slices -- no seg
    # sample_gan(model_directory=mri_slice_no_seg, output_directory='/home/abeers/Projects/PGGAN/MRI_Images/random_latent_images_no_seg', output_num=300, random_variation_num=None, batch_size=8, depth=7, channels=4, output_csv='random_mri_latents_no_seg.csv')


    # Generate Retina Slices
    # sample_gan(model_directory=retina_rgb_and_seg, output_directory='/home/abeers/Projects/PGGAN/Retinal_Images/random_latent_images/raw', output_num=300, random_variation_num=None, batch_size=1, depth=8, channels=4, output_csv='random_retina_latents.csv')

    # GEnerate Retina Slices with no Vessels
    # sample_gan(model_directory=retina_rgb_and_seg, output_directory='/home/abeers/Projects/PGGAN/Retinal_Images/random_latent_images_no_segs/', output_num=300, random_variation_num=None, batch_size=1, depth=8, channels=3, output_csv='random_retina_latents_no_seg.csv', latent_size=128)

    # retina_latent(input_model='retinal_model.pkl', output_model='retinal_model.pkl')

    # tumor_latent(input_npy='images.npy', input_csv='mri_latents.csv')

    find_projected(input_csv='mri_latents.csv', input_projected='projected_mri.npy')

    # find_projected()

    # encode_images(encoder_model_directory=encoding_retina_rgb_and_seg, gan_model_directory=retina_rgb_and_seg, output_directory='/home/abeers/Projects/PGGAN/Retinal_Images/inverse_gan_images/inverse_gan', images_directory='/home/abeers/Projects/PGGAN/Retinal_Images/raw/Posterior', mask_directory='/home/abeers/Projects/PGGAN/Retinal_Images/vessel_segmentation/Posterior', output_csv='retinal_latents.csv', depth=8, input_num=None, batch_size=8)


    # encode_images(images_directory='/mnt//QTIM_Databank/GBM_Slices_Alternate', encoder_model_directory=encoder_slice_with_seg, gan_model_directory=mri_slice_seg, output_directory=os.path.join(image_output_directory, 'Encoding_MRI_Slices_With_Seg'), output_csv='mri_latents.csv', only_channel=None, output_npy=['images_Alternate.npy', 'imageclasses_Alternate.npy', 'imagenames_Alternate.npy'], input_num=3500, channels=5, batch_size=10)

    # visualize_latents
