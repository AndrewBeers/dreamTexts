# From https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow

import tensorflow as tf
# from utils import save_images
# from utils import CelebA
import numpy as np
import scipy

from util import add_parameter
from ops import lrelu, conv2d, fully_connect, upscale, pixel_norm, avgpool2d, WScaleLayer, minibatch_state_concat

class PGGAN(object):

    # build model
    def __init__(self, **kwargs):

        # Training Parameters
        add_parameter(self, kwargs, 'batch_size', 16)
        add_parameter(self, kwargs, 'max_iterations', 32000)
        add_parameter(self, kwargs, 'learning_rate', 0.0001)
        add_parameter(self, kwargs, 'progressive_depth', 1)
        add_parameter(self, kwargs, 'transition', False)

        # Data Parameters
        self.data_In = data
        add_parameter(self, kwargs, 'samples_dir')
        add_parameter(self, kwargs, 'log_dir', './log')
        add_parameter(self, kwargs, 'input_model_path', None)
        add_parameter(self, kwargs, 'output_model_path', None)

        # Model Parameters
        add_parameter(self, kwargs, 'latent_size', 512)
        add_parameter(self, kwargs, 'max_filter', 1024)
        add_parameter(self, kwargs, 'channel', 3)

        # Derived Parameters
        self.log_vars = []
        self.output_size = 4 * pow(2, self.progressive_depth - 1)
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.latent = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')

    def get_filter_num(self, depth):

        # This will need to be a bit more complicated; see PGGAN paper.

        return min(self.max_filter / (2 **(depth * 1)), self.max_filter / 2)

    def generate(self, latent_var, progressive_depth=1, transition=False, alpha_transition=0.0):

        with tf.variable_scope('generator') as scope:

            convs = []

            convs += [tf.reshape(latent_var, [self.batch_size, 1, 1, self.get_filter_num(1)])]
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_h=4, k_w=4, d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')))

            convs += [tf.reshape(convs[-1], [self.batch_size, 4, 4, self.get_filter_num(1)])] # why necessary? --andrew
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_nf(1), d_w=1, d_h=1, name='gen_n_2_conv')))

            for i in range(progressive_depth - 1):

                if i == progressive_depth - 2 and transition: # redundant conditions? --andrew
                    #To RGB
                    # Don't totally understand this yet, diagram out --andrew
                    transition_conv = conv2d(convs[-1], output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(convs[-1].shape[1]))))

                convs += [pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(convs[-1].shape[1]))))]

            #To RGB
            convs += conv2d(convs[-1], output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))

            if progressive_depth == 1:
                return convs[-1]

            if transition:
                convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            return convs[-1]

    def discriminate(self, input_image, reuse=False, progressive_depth=1, transition=False, alpha_transition=0.01):

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

                convs += lrelu(conv2d(conv, output_dim=self.get_filter_num(progressive_depth - 2 - i), d_h=1, d_w=1, name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))
                convs[-1] = avgpool2d(convs[-1], 2)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=3, k_h=3, d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            conv = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))
            
            #for D
            output = tf.reshape(convs[-1], [self.batch_size, -1])
            output = fully_connect(output, output_size=1, scope='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def build_model(self):

        # Output functions
        self.fake_images = self.generate(self.latent, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, progressive_depth = self.pg, transition=self.transition, alpha_transition=self.alpha_transition)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True, progressive_depth= self.pg, transition=self.transition, alpha_transition=self.alpha_transition)

        # Loss functions
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
        # Also investigate more what's happening here --andrew
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits= self.discriminate(interpolates, reuse=True,  progressive_depth= self.pg, transition=self.transition, alpha_transition=self.alpha_transition)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # Some sort of norm from papers, check up on it. --andrew
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        # Update Loss functions..
        self.D_origin_loss = self.D_loss
        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        # Hmmm.. better way to do this? Or at least move to function.
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        # Huh
        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print variable.name, shape
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print "The total para of D", total_para

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print variable.name, shape
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print "The total para of G", total_para2

        #save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        print "d_vars", len(self.d_vars)
        print "g_vars", len(self.g_vars)

        print "self.d_vars_n_read", len(self.d_vars_n_read)
        print "self.g_vars_n_read", len(self.g_vars_n_read)

        print "d_vars_n_2_rgb", len(self.d_vars_n_2_rgb)
        print "g_vars_n_2_rgb", len(self.g_vars_n_2_rgb)

        # for n in self.d_vars:
        #     print n.name

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        print "self.g_d_w", len(self.g_d_w)

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)

        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)


    def train(self):

        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_transition_assign = self.alpha_transition.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0 , beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0 , beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.progressive_depth != 1 and self.progressive_depth != 7:

                if self.transition:
                    self.r_saver.restore(sess, self.input_model_path)
                    self.rgb_saver.restore(sess, self.input_model_path)

                else:
                    self.saver.restore(sess, self.input_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iterations:

                # optimization D
                n_critic = 1

                for i in range(n_critic):

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                    train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                    realbatch_array = CelebA.getShapeForData(train_list, resize_w=self.output_size)

                    if self.transition and self.progressive_depth != 0:

                        alpha = np.float(step) / self.max_iterations

                        low_realbatch_array = scipy.ndimage.zoom(realbatch_array, zoom=[1,0.5,0.5,1])
                        low_realbatch_array = scipy.ndimage.zoom(low_realbatch_array, zoom=[1, 2, 2, 1])
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.latent: sample_latent})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_transition_assign, feed_dict={step_pl: step})

                if step % 400 == 0:

                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.progressive_depth, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_real.png'.format(self.samples_dir, step))

                    if self.transition and self.progressive_depth != 0:

                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2], '{}/{:02d}_real_lower.png'.format(self.samples_dir, step))
                   
                    fake_image = sess.run(self.fake_images, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.png'.format(self.samples_dir, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.output_model_path)
                step += 1

            save_path = self.saver.save(sess, self.output_model_path)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()



if __name__ == '__main__':

    pass