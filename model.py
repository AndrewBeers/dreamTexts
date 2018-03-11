# From https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow

import tensorflow as tf
# from utils import save_images
# from utils import CelebA
import numpy as np
import scipy
import os

from util import add_parameter, save_images, save_image
from ops import lrelu, conv2d, fully_connect, upscale, downscale, pixel_norm, avgpool2d, WScaleLayer, minibatch_state_concat


class PGGAN(object):

    # build model
    def __init__(self, **kwargs):

        # Training Parameters
        add_parameter(self, kwargs, 'batch_size', 16)
        add_parameter(self, kwargs, 'max_iterations', 20000)
        add_parameter(self, kwargs, 'learning_rate', 0.0001)
        add_parameter(self, kwargs, 'progressive_depth', 1)
        add_parameter(self, kwargs, 'transition', False)
        add_parameter(self, kwargs, 'classes', True)

        # Data Parameters
        add_parameter(self, kwargs, 'training_data', None)
        add_parameter(self, kwargs, 'samples_dir', './samples')
        add_parameter(self, kwargs, 'log_dir', './log')
        add_parameter(self, kwargs, 'input_model_path', None)
        add_parameter(self, kwargs, 'output_model_path', None)

        # Model Parameters
        add_parameter(self, kwargs, 'progressive', True)
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'max_filter', 4096)
        add_parameter(self, kwargs, 'channels', 3)

        # Class Parameters
        add_parameter(self, kwargs, 'classify', None)
        add_parameter(self, kwargs, 'categorical_classes', None)
        add_parameter(self, kwargs, 'continuous_classes', None)

        # Test Parameters
        add_parameter(self, kwargs, 'testing', False)

        # Reverse GAN Parameters
        add_parameter(self, kwargs, 'reverse', False)
        add_parameter(self, kwargs, 'reverse_cost', 'dice')

        # Encoder Parameters
        add_parameter(self, kwargs, 'encoding', False)
        add_parameter(self, kwargs, 'input_encoder_model_path', None)
        add_parameter(self, kwargs, 'output_encoder_model_path', None)
        add_parameter(self, kwargs, 'encoder_log_dir', None)
        add_parameter(self, kwargs, 'only_channel', None)

        # Save Parameters
        add_parameter(self, kwargs, 'save_model_path', None)

        if self.progressive_depth >= 8 and not self.encoding and not self.testing and self.save_model_path is None:
            self.batch_size = 4

        # if self.testing:
        #     self.batch_size = 8

        # Derived class variables -- perhaps excessive.
        self.class_num = 0
        if self.continuous_classes is not None:
            self.contiuous_latent = tf.placeholder(tf.float32, [self.batch_size, self.continuous_classes])
            self.class_num += self.continuous_classes
        if self.categorical_classes is not None:
            self.class_num += self.categorical_classes

        # This is all messed up.
        self.categorical_latent = tf.placeholder(tf.float32, [self.batch_size, self.categorical_classes])
        self.true_categorical_latent = tf.placeholder(tf.float32, [self.batch_size, self.categorical_classes])

        if self.class_num > 0:
            self.latent_size += self.class_num
            self.base_latent = tf.placeholder(tf.float32, [self.batch_size, self.latent_size])
            self.latent = tf.concat([self.categorical_latent, self.base_latent], axis=1) # Has error, will need fixing
        else:
            self.latent = tf.placeholder(tf.float32, [self.batch_size, self.latent_size])

        # Derived Parameters
        self.log_vars = []
        self.output_size = pow(2, self.progressive_depth + 1)
        self.zoom_level = self.progressive_depth
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channels], name='discriminator/discriminator_input')
        self.seg_images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, 1])
        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')


    def get_filter_num(self, depth):

        # This will need to be a bit more complicated; see PGGAN paper.
        if depth == 8:
            return 16
        if min(self.max_filter / (2 **(depth)), 128) <= 32:
            return 16
        else:
            return min(self.max_filter / (2 **(depth)), 128)

    def generate(self, latent_var, progressive_depth=1, transition=False, alpha_transition=0.0):

        if self.encoding:
            transition = False

        with tf.variable_scope('generator') as scope:

            convs = []

            if self.classify is None:
                convs += [tf.reshape(latent_var, [self.batch_size, 1, 1, self.latent_size])]
            else:
                convs += [tf.reshape(latent_var, [self.batch_size, 1, 1, self.latent_size + self.class_num])]

            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_h=4, k_w=4, d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')))

            convs += [tf.reshape(convs[-1], [self.batch_size, 4, 4, self.get_filter_num(1)])] # why necessary? --andrew
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), d_w=1, d_h=1, name='gen_n_2_conv')))

            for i in range(progressive_depth - 1):

                if i == progressive_depth - 2 and transition: # redundant conditions? --andrew
                    #To RGB
                    # Don't totally understand this yet, diagram out --andrew
                    transition_conv = conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(convs[-1].shape[1]))))

                convs += [pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(convs[-1].shape[1]))))]


            #To RGB
            convs += [conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))]

            if transition:
                convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            return convs[-1]

    def discriminate(self, input_image, reuse=False, progressive_depth=1, transition=False, alpha_transition=0.01, input_classes=None):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            if transition:
                transition_conv = avgpool2d(input_image)
                transition_conv = lrelu(conv2d(transition_conv, output_dim= self.get_filter_num(progressive_depth - 2), k_w=1, k_h=1, d_h=1, d_w=1, name='dis_y_rgb_conv_{}'.format(transition_conv.shape[1])))

            convs = []

            # fromRGB
            convs += [lrelu(conv2d(input_image, output_dim=self.get_filter_num(progressive_depth - 1), k_w=1, k_h=1, d_w=1, d_h=1, name='dis_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(progressive_depth - 1):

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 1 - i), d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 2 - i), d_h=1, d_w=1, name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = avgpool2d(convs[-1], 2)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=3, k_h=3, d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            if False:
                convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))
        

            #for D
            output = tf.reshape(convs[-1], [self.batch_size, -1])

            if self.classify is None:
                discriminate_output = fully_connect(output, output_size=1, scope='dis_n_fully')
                return None, None, tf.nn.sigmoid(discriminate_output), discriminate_output
                # return None, None, tf.nn.sigmoid(output), output
            else:
                # discriminate_output = fully_connect(output, output_size=128, scope='dis_n_fully_1')
                discriminate_output = tf.concat([output, input_classes], axis=1)
                discriminate_output = fully_connect(discriminate_output, output_size=1, scope='dis_n_fully')
                return None, None, tf.nn.sigmoid(discriminate_output), discriminate_output
                # classify_output = fully_connect(output, output_size=128, scope='gen_n_fully_class_1')
                # classify_output = fully_connect(output, output_size=self.class_num, scope='gen_n_fully_class_2')
                # return tf.nn.softmax(classify_output), classify_output, tf.nn.sigmoid(discriminate_output), discriminate_output


    def build_model(self):

        # Output functions
        self.fake_images = self.generate(self.latent, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        
        if not self.encoding:

            self.Q_generated_real, _, _, self.D_pro_logits = self.discriminate(self.images, reuse=False, progressive_depth = self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, input_classes=self.true_categorical_latent)
            self.Q_generated_fake, _, _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True, progressive_depth= self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, input_classes=self.categorical_latent)

            # Loss functions
            self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
            self.G_loss = -tf.reduce_mean(self.G_pro_logits)

            # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
            # Also investigate more what's happening here --andrew
            self.differences = self.fake_images - self.images
            self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolates = self.images + (self.alpha * self.differences)
            _, _, _, discri_logits= self.discriminate(interpolates, reuse=True,  progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, input_classes=self.true_categorical_latent)
            gradients = tf.gradients(discri_logits, [interpolates])[0]

            # Some sort of norm from papers, check up on it. --andrew
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
            self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            tf.summary.scalar("gp_loss", self.gradient_penalty)

            # Update Loss functions..
            self.D_origin_loss = self.D_loss
            self.D_loss += 10 * self.gradient_penalty
            self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

            # if self.classify is not None:
            #     self.Q_loss_real = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_generated_real + 1e-8) * self.true_categorical_latent, axis=1)) + tf.reduce_mean(-tf.reduce_sum(tf.log(self.true_categorical_latent + 1e-8) * self.true_categorical_latent, axis=1))
            #     self.Q_loss_fake = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_generated_fake + 1e-8) * self.categorical_latent, axis=1)) + tf.reduce_mean(-tf.reduce_sum(tf.log(self.categorical_latent + 1e-8) * self.categorical_latent, axis=1))
            #     self.Q_loss = Q_loss_real + Q_loss_fake

            self.log_vars.append(("generator_loss", self.G_loss))
            self.log_vars.append(("discriminator_loss", self.D_loss))
            # self.log_vars.append(("class_loss", self.Q_loss))

        # Classifier Tools
        if self.classify is not None:

            self.cat_real = self.gan_classify(self.images, reuse=False, progressive_depth = self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition,)
            self.cat_fake = self.gan_classify(self.fake_images, reuse=True, progressive_depth = self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)

            self.loss_c_f = tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_fake, labels=self.categorical_latent)
            self.loss_c_f_sum = tf.summary.scalar("categorical_loss_c_fake", tf.reduce_mean(self.loss_c_f))
            self.loss_c_r = tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_real, labels=self.true_categorical_latent)
            self.loss_c_r_sum = tf.summary.scalar("categorical_loss_c_real", tf.reduce_mean(self.loss_c_r))
            self.loss_c = (self.loss_c_r + self.loss_c_f) / 2
            self.loss_c_sum = tf.summary.scalar("categorical_loss_c", tf.reduce_mean(self.loss_c))

        # Data Loading Tools
        self.low_images = upscale(downscale(self.images, 2), 2)
        self.real_images = self.alpha_transition * self.images + (1 - self.alpha_transition) * self.low_images

        # Encoder Tools
        if self.only_channel is not None:
            self.decoded_latent = self.encode(self.seg_images, reuse=False, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        else:
            self.decoded_latent = self.encode(self.images, reuse=False, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        self.E_loss = tf.reduce_sum(tf.pow(self.latent - self.decoded_latent, 2), [0, 1])

        # Hmmm.. better way to do this? Or at least move to function.
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'enc' in var.name]
        self.c_vars = [var for var in t_vars if 'class' in var.name]

        # save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]
        self.e_vars_n = [var for var in self.e_vars if 'enc_n' in var.name]
        self.c_vars_n = [var for var in self.c_vars if 'class_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]
        self.e_vars_n_read = [var for var in self.e_vars_n if '{}'.format(self.output_size) not in var.name]
        self.c_vars_n_read = [var for var in self.c_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]
        self.e_vars_n_2 = [var for var in self.e_vars if 'enc_y_rgb_conv' in var.name]
        self.c_vars_n_2 = [var for var in self.c_vars if 'class_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.e_vars_n_2_rgb = [var for var in self.e_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.c_vars_n_2_rgb = [var for var in self.c_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        if self.classify is not None:
            self.saver = tf.train.Saver(self.d_vars + self.g_vars + self.c_vars)
            self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read + self.c_vars_n_read)
            if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb + self.c_vars_n_2_rgb):
                self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb + self.c_vars_n_2_rgb)
        else:
            self.saver = tf.train.Saver(self.d_vars + self.g_vars)
            self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)
            if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
                self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for layer in self.d_vars + self.g_vars:
            print layer

        self.encoder_saver = tf.train.Saver(self.e_vars)
        self.encoder_r_saver = tf.train.Saver(self.e_vars_n_read)

        # if len(self.e_vars_n_2_rgb):
            # self.encoder_rgb_saver = tf.train.Saver(self.e_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        # # Reverse GAN Tools
        # if not self.encoding:
        #     self.reverse_z = tf.placeholder(tf.float32, [None, self.latent_size], name='reverse_z')
        #     self.imposter_images = self.imposter_generate(self.reverse_z, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)

        #     if self.reverse_cost == 'dice':

        #         smooth = 1e-5
        #         axis = [1, 2]
        #         output, target = self.fake_images[..., -1], self.imposter_images[..., -1]
        #         inse = tf.reduce_sum(output * target, axis=axis)
        #         l = tf.reduce_sum(output * output, axis=axis)
        #         r = tf.reduce_sum(target * target, axis=axis)
        #         dice = (2. * inse + smooth) / (l + r + smooth)
        #         self.reverse_loss = tf.reduce_mean(dice)

        #     elif self.reverse_cost == 'mse':

        #         self.reverse_loss_sum = tf.reduce_sum(tf.pow(self.fake_images[..., -1] - self.imposter_images[..., -1], 2), 0)
        #         self.reverse_loss = tf.reduce_mean(self.reverse_loss_sum)

        #     self.reverse_gradients = tf.gradients(self.reverse_loss, self.reverse_z)

    def train_cgan(self):

        # Create fade-in (transition) parameters.
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_transition_assign = self.alpha_transition.assign(step_pl / self.max_iterations)

        # Create Optimizers
        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)
        opti_C_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.loss_c_f, var_list=self.g_vars)
        opti_C = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.loss_c, var_list=self.c_vars)
        # opti_Q = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            # self.Q_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            # No idea what the saving systems is like. TODO investigate --andrew.
            # I don't think you need to save and reload models if you create a crazy
            # system where you're only optimizing certain outputs/cost functions at
            # any one time.
            if self.progressive_depth != 1:

                if self.transition:
                    self.r_saver.restore(sess, self.input_model_path)
                    self.rgb_saver.restore(sess, self.input_model_path)
                else:
                    self.saver.restore(sess, self.input_model_path)

            sample_categories = np.zeros((self.batch_size, 2), dtype=float)

            step = 0
            batch_num = 0
            while step <= self.max_iterations:

                n_critic = 1

                # Update Discriminator
                for i in range(n_critic):

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                    # sample_categories = np.concatenate(np.random.randint(low=0, high=2, size=[self.batch_size, 1]), np.random.randint(low=0, high=3, size=[self.batch_size, 1]), axis=1)

                    for i in xrange(self.batch_size):
                        sample_categories[i, :] = np.array([np.random.randint(low=0, high=2), np.random.randint(low=0, high=3)])

                    if self.classes:
                        realbatch_array, realclasses_array = self.training_data.get_next_batch_classed(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size, infogan=True)
                    else:
                        realbatch_array, realclasses_array = self.training_data.get_next_batch(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)

                    if self.transition:
                        
                        realbatch_array = sess.run(self.real_images, feed_dict={self.images: realbatch_array})

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.base_latent: sample_latent, self.categorical_latent: sample_categories, self.true_categorical_latent: realclasses_array})
                    batch_num += 1

                # Update Generator
                sess.run(opti_G, feed_dict={self.base_latent: sample_latent, self.categorical_latent: sample_categories})

                if step % 2 == 1:
                    sess.run(opti_C, feed_dict={self.images: realbatch_array, self.base_latent: sample_latent, self.categorical_latent: sample_categories, self.true_categorical_latent: realclasses_array})
                    sess.run(opti_C_G, feed_dict={self.images: realbatch_array, self.base_latent: sample_latent, self.categorical_latent: sample_categories, self.true_categorical_latent: realclasses_array})

                # summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                # summary_writer.add_summary(summary_str, step)

                # the alpha of fake_in process
                sess.run(alpha_transition_assign, feed_dict={step_pl: step})

                if step % 40 == 0:
                    D_loss, G_loss, D_origin_loss, C_loss, C_G_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.loss_c, self.loss_c_f, self.alpha_transition], feed_dict={self.images: realbatch_array, self.base_latent: sample_latent, self.categorical_latent: sample_categories, self.true_categorical_latent: realclasses_array})
                    print("PG %d, step %d: D loss=%.4f G loss=%.4f, D_or loss=%.4f, C loss=%.4f, C_G_loss=%.4f, opt_alpha_tra=%.4f" % (self.progressive_depth, step, D_loss, G_loss, D_origin_loss, np.mean(C_loss), np.mean(C_G_loss), alpha_tra))

                if step % 400 == 0:

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_real.png'.format(self.samples_dir, step))

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                    for side in [0, 1]:
                        for disease in [0, 1, 2]:
                            for i in xrange(self.batch_size):
                                sample_categories[i, :] = np.array([side, disease])
                            fake_image = sess.run(self.fake_images, feed_dict={self.images: realbatch_array, self.base_latent: sample_latent, self.categorical_latent: sample_categories, self.true_categorical_latent: realclasses_array})
                            fake_image = np.clip(fake_image, -1, 1)
                            save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train_{:02d}_{:02d}.png'.format(self.samples_dir, step, side, disease))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.output_model_path)
                step += 1

            save_path = self.saver.save(sess, self.output_model_path)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()

    def train(self):

        # Create fade-in (transition) parameters.
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_transition_assign = self.alpha_transition.assign(step_pl / self.max_iterations)

        # Create Optimizers
        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            # No idea what the saving systems is like. TODO investigate --andrew.
            # I don't think you need to save and reload models if you create a crazy
            # system where you're only optimizing certain outputs/cost functions at
            # any one time.
            if self.progressive_depth != 1:

                if self.transition:
                    self.r_saver.restore(sess, self.input_model_path)
                    self.rgb_saver.restore(sess, self.input_model_path)
                else:
                    self.saver.restore(sess, self.input_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iterations:

                n_critic = 1

                # Update Discriminator
                for i in range(n_critic):

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])

                    if self.classes:
                        realbatch_array, _ = self.training_data.get_next_batch_classed(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)
                    else:
                        realbatch_array, _ = self.training_data.get_next_batch(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)

                    realbatch_array = realbatch_array[...,0:4]

                    if self.transition:
                        
                        realbatch_array = sess.run(self.real_images, feed_dict={self.images: realbatch_array})

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    batch_num += 1

                # Update Generator
                sess.run(opti_G, feed_dict={self.latent: sample_latent})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                summary_writer.add_summary(summary_str, step)

                # the alpha of fake_in process
                sess.run(alpha_transition_assign, feed_dict={step_pl: step})

                if step % 40 == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_transition], feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.progressive_depth, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                if step % 400 == 0:

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_real.png'.format(self.samples_dir, step))

                    fake_image = sess.run(self.fake_images, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.png'.format(self.samples_dir, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.output_model_path)
                step += 1

            save_path = self.saver.save(sess, self.output_model_path)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()

    def test_model(self, output_directory, output_num, output_format='png', input_latent=None):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)

            # No idea what the saving systems is like. TODO investigate --andrew.
            # I don't think you need to save and reload models if you create a crazy
            # system where you're only optimizing certain outputs/cost functions at
            # any one time.
            self.saver.restore(sess, self.input_model_path)

            mode = 'axis'

            if input_latent is not None:
                fake_image = sess.run(self.fake_images, feed_dict={self.latent: input_latent})
                fake_image = np.clip(fake_image, -1, 1)
                save_images(fake_image, [2, 4], '{}/{:02d}_test.{}'.format(output_directory, self.progressive_depth, output_format))      

            else:
                if mode == 'random':
                    for idx in xrange(output_num):
                        print idx
                        sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                        fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                        fake_image = np.clip(fake_image, -1, 1)
                        save_image(fake_image, '{}/{:02d}_test.{}'.format(output_directory, idx, output_format))

                if mode == 'slerp':
                    sample_latent1 = np.random.normal(size=[self.latent_size])
                    sample_latent2 = np.random.normal(size=[self.latent_size])
                    for idx in xrange(output_num):
                        sample_latent = slerp(float(idx) / output_num, sample_latent1, sample_latent2)[np.newaxis]
                        print np.max(sample_latent) - np.min(sample_latent)
                        fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                        fake_image = np.clip(fake_image, -1, 1)
                        save_image(fake_image, '{}/{:02d}_test.{}'.format(output_directory, idx, output_format))

                if mode == 'tozero':
                    zeros = np.zeros((self.latent_size))
                    sample_latent1 = np.random.normal(size=[self.latent_size])
                    sample_latent2 = np.random.normal(size=[self.latent_size])
                    for idx in xrange(output_num):
                        if idx < output_num / 2:
                            sample_latent = lerp(float(idx) / output_num / 2, sample_latent1, zeros)[np.newaxis]
                        else:
                            sample_latent = lerp(float(idx-output_num/2) / output_num / 2, zeros, sample_latent2)[np.newaxis]
                        print idx
                        fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                        fake_image = np.clip(fake_image, -1, 1)
                        save_image(fake_image, '{}/{:02d}_test.{}'.format(output_directory, idx, output_format))

                if mode == 'points':
                    point_list = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]]
                    count = 0
                    for point_idx, point in enumerate(point_list[:-1]):
                        sample_latent1 = np.array(point)
                        sample_latent2 = np.array(point_list[point_idx + 1])
                        for idx in xrange(output_num / (len(point_list) - 1)):
                            sample_latent = lerp(float(idx) / (output_num / (len(point_list) - 1)), sample_latent1, sample_latent2)[np.newaxis]
                            print np.max(sample_latent) - np.min(sample_latent)
                            fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                            fake_image = np.clip(fake_image, -1, 1)
                            save_image(fake_image, '{}/{:02d}_test.png'.format(output_directory, count))
                            count += 1

                if mode == 'axis':
                    axis = 0
                    axis_batch = 6
                    sample_latent1 = np.random.normal(size=[self.batch_size, self.latent_size])
                    sample_latent2 = np.random.normal(size=[self.batch_size, self.latent_size])
                    sample_latent2[:, axis] = 3 * np.ones((self.batch_size))
                    sample_latent1[:, axis] = -3 * np.ones((self.batch_size))
                    for idx in xrange(output_num):
                        sample_latent = batch_slerp(float(idx) / output_num, sample_latent1, sample_latent2)
                        print sample_latent1.shape
                        print np.max(sample_latent) - np.min(sample_latent)
                        print sample_latent.shape
                        fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                        fake_image = np.clip(fake_image, -1, 1)
                        save_images(fake_image, [2, 4], '{}/{:02d}_test.{}'.format(output_directory, idx, output_format))


                if mode == 'grid':
                    x = np.linspace(-3, 3, 100)
                    y = np.linspace(-3, 3, 100)
                    for x_idx in x:
                        for y_idx in y:
                            print x_idx, y_idx
                            sample_latent = np.array([x_idx,y_idx])[np.newaxis]
                            fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                            fake_image = np.clip(fake_image, -1, 1)
                            save_image(fake_image, '{}/{}_test.png'.format(output_directory, str(x_idx) + '_' + str(y_idx)))                        

        tf.reset_default_graph()    

    def imposter_generate(self, latent_var, progressive_depth=1, transition=False, alpha_transition=0.0):

        with tf.variable_scope('generator') as scope:

            scope.reuse_variables()

            convs = []

            convs += [tf.reshape(latent_var, [self.batch_size, 1, 1, self.latent_size])]
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_h=4, k_w=4, d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')))

            convs += [tf.reshape(convs[-1], [self.batch_size, 4, 4, self.get_filter_num(1)])] # why necessary? --andrew
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), d_w=1, d_h=1, name='gen_n_2_conv')))

            for i in range(progressive_depth - 1):

                if i == progressive_depth - 2 and transition: # redundant conditions? --andrew
                    #To RGB
                    # Don't totally understand this yet, diagram out --andrew
                    transition_conv = conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(convs[-1].shape[1]))))

                convs += [pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(convs[-1].shape[1]))))]


            #To RGB
            convs += [conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))]

            if progressive_depth == 1:
                return convs[-1]

            if transition:
                if self.encoding:
                    pass
                else:
                    convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            return convs[-1]

    def encode(self, input_image, reuse=False, progressive_depth=1, transition=False, alpha_transition=0.01):

        with tf.variable_scope("encoder") as scope:

            if reuse == True:
                scope.reuse_variables()

            if transition:
                transition_conv = avgpool2d(input_image)
                transition_conv = lrelu(conv2d(transition_conv, output_dim= self.get_filter_num(progressive_depth - 2), k_w=1, k_h=1, d_h=1, d_w=1, name='enc_y_rgb_conv_{}'.format(transition_conv.shape[1])))

            convs = []

            # fromRGB
            convs += [lrelu(conv2d(input_image, output_dim=self.get_filter_num(progressive_depth - 1), k_w=1, k_h=1, d_w=1, d_h=1, name='enc_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(progressive_depth - 1):

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 1 - i), d_h=1, d_w=1, name='enc_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 2 - i), d_h=1, d_w=1, name='enc_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = avgpool2d(convs[-1], 2)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=3, k_h=3, d_h=1, d_w=1, name='enc_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            conv = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='enc_n_conv_2_{}'.format(convs[-1].shape[1])))
            
            #for D
            output = tf.reshape(convs[-1], [self.batch_size, -1])
            output = fully_connect(output, output_size=self.latent_size, scope='enc_n_fully')

            return output

    def gan_classify(self, input_image, reuse=False, progressive_depth=1, transition=False, alpha_transition=0.01, filter_downscale=2):

        with tf.variable_scope("classifier") as scope:

            if reuse == True:
                scope.reuse_variables()

            if transition:
                transition_conv = avgpool2d(input_image)
                transition_conv = lrelu(conv2d(transition_conv, output_dim= self.get_filter_num(progressive_depth - 2)/filter_downscale, k_w=1, k_h=1, d_h=1, d_w=1, name='class_y_rgb_conv_{}'.format(transition_conv.shape[1])))

            convs = []

            # fromRGB
            convs += [lrelu(conv2d(input_image, output_dim=self.get_filter_num(progressive_depth - 1)/filter_downscale, k_w=1, k_h=1, d_w=1, d_h=1, name='class_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(progressive_depth - 1):

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 1 - i)/filter_downscale, d_h=1, d_w=1, name='class_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(progressive_depth - 2 - i)/filter_downscale, d_h=1, d_w=1, name='class_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = avgpool2d(convs[-1], 2)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1)/filter_downscale, k_w=3, k_h=3, d_h=1, d_w=1, name='class_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            conv = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1)/filter_downscale, k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='class_n_conv_2_{}'.format(convs[-1].shape[1])))
            
            #for D
            output = tf.reshape(convs[-1], [self.batch_size, -1])
            output = fully_connect(output, output_size=self.categorical_classes, scope='class_n_fully')

            return output       

    def fit_latent_vector(self, output_directory, initialized_latent=None):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        np.set_printoptions(suppress=True)

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)

            self.saver.restore(sess, self.input_model_path)

            # Showing the progress on precision when the latent vector is known.
            precision_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
           
            if initialized_latent is None:
                # imposter_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                imposter_latent = np.zeros(shape=(self.batch_size, self.latent_size))
            else:
                imposter_latent = initialized_latent

            # Check on difference between these modes.
            # configurations = ["no_clipping", "standard_clipping", "stochastic_clipping"]
           
            # Options for using preloaded images or not 
            if self.training_data is not None:        
                target_image = self.training_data.get_next_batch(batch_num=0, zoom_level=self.zoom_level, batch_size=self.batch_size)
            else:
                sample_latent = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_size)) ## input  
                target_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})

            imposter_image_base = sess.run(self.imposter_images, feed_dict={self.reverse_z: imposter_latent})
            imposter_image = np.copy(imposter_image_base)

            reverse_accuracy_matrix = np.zeros((self.batch_size, len(precision_levels)))
            accuracy_directory = os.path.join(output_directory, "accuracy/")                 
            if not os.path.exists(accuracy_directory):
                os.makedirs(accuracy_directory)

            target_filepath = os.path.join(output_directory, "visualization_target.jpg")
            save_images(target_image, [2, 4], target_filepath)
            imposter_filepath = os.path.join(output_directory, "visualization_imposter_init.jpg")
            save_images(np.clip(imposter_image, -1, 1), [2, 4], imposter_filepath)
   
            if self.training_data is None:
                z_loss = []
                imposter_latent_loss = np.sum(np.power(sample_latent - imposter_latent, 2), 0)
                imposter_latent_loss = np.mean(imposter_latent_loss)
                z_loss.append(imposter_latent_loss)
            else:
                recon_zs = []

            phi_loss = []
            initial_phi_loss = np.sum(np.power(target_image - imposter_image, 2), 0)  # sum over batch
            initial_phi_loss = np.mean(initial_phi_loss)             
            phi_loss.append(initial_phi_loss)

            learning_rate = 1
            iterations = 1000
            
            for n_iter in np.arange(iterations):

                # Halve learning rate every 50000 iterations.
                # Why not just use Adam tho?
                if n_iter % 50000 == 0 and n_iter > 0:
                    learning_rate /= 2.

                if n_iter > 0:
                    phi_loss.append(reverse_loss)

                    # If you know the real latent, keep track of accuracy over time.
                    if self.training_data is None:
                        curr_z_loss = np.mean(np.power(imposter_latent - sample_latent, 2), 1)
                        z_loss.append(np.mean(curr_z_loss)) 

                        # Keep track of accuracy in the latent space up to a certain degree of precision.
                        for p in np.arange(len(precision_levels)):
                            curr_z_loss_p = curr_z_loss
                            prec_level = precision_levels[p]
                            d = np.where(curr_z_loss_p < prec_level)
                            reverse_accuracy_matrix[d, p] = 1

                        # Stop when all precision levels are reached (this is kind of wild)
                        if np.sum(reverse_accuracy_matrix) == self.batch_size * len(precision_levels):
                            break                       

                [reverse_loss, reverse_gradients, imposter_image] = sess.run([self.reverse_loss, self.reverse_gradients, self.imposter_images], feed_dict={self.fake_images: target_image, self.reverse_z: imposter_latent})

                reverse_gradients = np.asarray(reverse_gradients[0])
                # print imposter_latent

                if n_iter % 200 == 0:
                    if self.training_data is None:
                        print(n_iter, reverse_loss, np.mean(np.power(imposter_latent - sample_latent, 2)))
                    else:
                        print(n_iter, reverse_loss)                             
                        output_filepath = os.path.join(output_directory, "visualization_imposter" + str(n_iter) + ".jpg")
                        save_images(np.clip(imposter_image, -1, 1), [2, 4], output_filepath)

                imposter_latent = imposter_latent - learning_rate * reverse_gradients            
                     
                # # Stochastic Clipping
                # for j in range(self.batch_size):  
                #     edge1 = np.where(imposter_latent[j] >= 4)[0] #1
                #     edge2 = np.where(imposter_latent[j] <= -4)[0] #1

                #     if edge1.shape[0] > 0:
                #         rand_el1 = np.random.normal(size=(1, edge1.shape[0])) 
                #         imposter_latent[j,edge1] = rand_el1
                #     if edge2.shape[0] > 0:
                #         rand_el2 = np.random.normal(size=(1, edge2.shape[0]))                            
                #         imposter_latent[j,edge2] = rand_el2

        tf.reset_default_graph()  
        return imposter_latent


            # if self.training_data is not None:
            #     for j in np.arange(self.batch_size):
            #         recon_zs.append(imposter_latent[j])
            #     recon_z_info = {'recon_zs':recon_zs}
            #     file_name = output_directory + "recon_zs.pickle"
            #     with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
            #         pickle.dump(recon_z_info, handle, protocol=pickle.HIGHEST_PROTOCOL)               

            # if self.training_data is not None:
            #     information = {'phi_loss': phi_loss, 
            #                   'z_loss':z_loss,
            #                   'sample_latent': sample_latent,
            #                    'imposter_latent': imposter_latent}
            # else:
            #     information = {'phi_loss': phi_loss,
            #                   'imposter_latent': imposter_latent}

            # file_name = output_directory + "GAN_loss.pickle"
            # with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
            #     pickle.dump(information, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # accuracy_information = {'accuracy_stats': reverse_accuracy_matrix}
            # file_name = output_directory + "accuracy_stats.pickle"
            # with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
            #     pickle.dump(accuracy_information, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_encoder(self):

        # Create Optimizers
        opti_encoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.E_loss, var_list=self.e_vars)

        # Create fade-in (transition) parameters.
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_transition_assign = self.alpha_transition.assign(step_pl / self.max_iterations)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.encoder_log_dir, sess.graph)

            # No idea what the saving systems is like. TODO investigate --andrew.
            # I don't think you need to save and reload models if you create a crazy
            # system where you're only optimizing certain outputs/cost functions at
            # any one time.
            if self.progressive_depth != 1 and self.progressive:

                if self.transition:
                    # self.r_saver.restore(sess, self.input_model_path)
                    # self.rgb_saver.restore(sess, self.input_model_path)
                    self.encoder_r_saver.restore(sess, self.input_encoder_model_path)
                    self.encoder_rgb_saver.restore(sess, self.input_encoder_model_path)
                    self.saver.restore(sess, self.input_model_path)
                else:
                    self.encoder_saver.restore(sess, self.input_encoder_model_path)
                    self.saver.restore(sess, self.input_model_path)

            else:
                self.saver.restore(sess, self.input_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iterations:

                sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])

                generated_images = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})

                if self.transition:
                        
                    generated_images = sess.run(self.real_images, feed_dict={self.images: generated_images})

                if self.only_channel is not None: # Messssyyyyyy
                    generated_images = generated_images[...,self.only_channel][...,np.newaxis]
                    input_images = self.seg_images
                else:
                    input_images = self.images

                sess.run(opti_encoder, feed_dict={input_images: generated_images, self.latent: sample_latent})
                sess.run(alpha_transition_assign, feed_dict={step_pl: step})

                if step % 40 == 0:
                    E_loss = sess.run([self.E_loss], feed_dict={input_images: generated_images, self.latent: sample_latent})
                    print("PG %d, step %d: E loss=%.7f" % (self.progressive_depth, step, E_loss[0]))

                if step % 400 == 0:

                    true_images = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})

                    if self.only_channel is not None:
                        generated_latent = sess.run(self.decoded_latent, feed_dict={input_images: true_images[...,self.only_channel][...,np.newaxis]})
                    else:
                        generated_latent = sess.run(self.decoded_latent, feed_dict={input_images: true_images})

                    generated_images = sess.run(self.fake_images, feed_dict={self.latent: generated_latent})

                    true_images[self.batch_size/2:] = generated_images[0:self.batch_size/2]
                    true_images = np.clip(true_images, -1, 1)
                    save_images(true_images[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_decoded.png'.format(self.samples_dir, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.encoder_saver.save(sess, self.output_encoder_model_path)
                step += 1

            save_path = self.encoder_saver.save(sess, self.output_encoder_model_path)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()

        return

    def test_encoder(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        np.set_printoptions(suppress=True)

        with tf.Session(config=config) as sess:

            sess.run(init)

            self.saver.restore(sess, self.input_model_path)
            self.encoder_saver.restore(sess, self.input_encoder_model_path)

            for batch_num in xrange(self.training_data.image_num / self.batch_size):

                target_image = self.training_data.get_next_batch(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)

                if self.only_channel is None:
                    generated_latent = sess.run(self.decoded_latent, feed_dict={self.images: target_image})
                else:
                    generated_latent = sess.run(self.decoded_latent, feed_dict={self.seg_images: target_image[...,self.only_channel][...,np.newaxis]})

                generated_images = sess.run(self.fake_images, feed_dict={self.latent: generated_latent})

                image_outputs = np.concatenate((target_image, generated_images), axis=0)
                image_outputs = np.clip(image_outputs, -1, 1)
                save_images(image_outputs, [2, self.batch_size], '{}/{:02d}_decoded.png'.format(self.samples_dir, batch_num))

    def save_model(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        builder = tf.saved_model.builder.SavedModelBuilder(self.save_model_path)

        with tf.Session(config=config) as sess:

            sess.run(init)

            if self.transition:
                self.r_saver.restore(sess, self.input_model_path)
                self.rgb_saver.restore(sess, self.input_model_path)
            else:
                self.saver.restore(sess, self.input_model_path)

            builder.add_meta_graph_and_variables(sess, ['rop_masks'])

        builder.save()
        tf.reset_default_graph()


def slerp(val, low, high):

    print val
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def batch_slerp(val, low, high):

    print val
    output_batch = np.zeros_like(low)

    """Spherical interpolation. val has a range of 0 to 1."""
    for idx, vals in enumerate(zip(low, high)):
        low, high = vals
        if val <= 0:
            output_batch[idx, :] = low
        elif val >= 1:
            output_batch[idx, :] = high
        elif np.allclose(low, high):
            output_batch[idx, :] = low
        omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
        so = np.sin(omega)
        output_batch[idx, :] = np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

    print 'output batch shape', output_batch.shape
    return output_batch

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

if __name__ == '__main__':

    pass