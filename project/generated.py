import tensorflow as tf
import numpy as np
import vgg19
import copy
import scipy
import skimage


class neural_style(object):
    def __init__(self,
                content_layers,
                style_layers,
                alpha,
                beta,
                weights):
        self.vgg = vgg19.Vgg19()
        self.images = tf.placeholder("float", [1, 224, 224, 3])
        self.vgg.build(self.images)
        self.alpha = alpha
        self.beta = beta
        self.lr = 2
        self.style_weights = weights
        self.content_layers = content_layers.split(',')
        self.style_layers = style_layers.split(',')
        self.layers = {'conv1_1' : self.vgg.conv1_1,
                'conv2_1' : self.vgg.conv2_1,
                'conv3_1' : self.vgg.conv3_1,
                'conv4_1' : self.vgg.conv4_1,
                'conv5_1' : self.vgg.conv5_1}
        self.count = 0

    def run(self, content_image, style_image):
        b = self.beta
        a = self.alpha
        iterations = 10
        self.output = np.copy(style_image)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            content_loss = self.get_cont_loss(content_image)
            style_loss = self.get_style_loss(style_image)
            total_loss = b * tf.reduce_sum(content_loss) + a * tf.reduce_sum(style_loss)
            print("loss computed\n")
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(total_loss, options = {
                    'maxiter': iterations,
                    'disp': 2}, method = 'L-BFGS-B')
            print("minimizing\n")
            optimizer.minimize(sess, step_callback = self.step_call)
        return total_loss


    def get_cont_loss(self, content):
        losses = []
        layers = [self.layers[i] for i in self.content_layers]
        with tf.Session() as sess:
            P = sess.run(layers, {self.images: self.output})
            F = sess.run(layers, {self.images: content})
        for i in range(len(layers)):
            loss = 0.5 * tf.reduce_sum(tf.squared_difference(F[i], P[i]))
            losses.append(loss)
        return losses

    def get_style_loss(self, style):
        losses = []
        layers = [self.layers[i] for i in self.style_layers]
        with tf.Session() as sess:
            feature_s = sess.run(layers, {self.images: style})
            feature_o = sess.run(layers, {self.images: self.output})
        for i in range(len(layers)):
            shape = feature_s[i].shape
            M,N = shape[1]*shape[2], shape[3]
            vect_s = feature_s[i].reshape(M,N)
            vect_o = feature_o[i].reshape(M,N)
            A = tf.matmul(vect_s, vect_s, transpose_a = True)
            G = tf.matmul(vect_o, vect_o, transpose_a = True)
            loss = 1/(4*N^2*M^2) * tf.reduce_sum(tf.squared_difference(A, G))
            loss = tf.multiply(self.style_weights, loss)
            losses.append(loss)
        return losses

    def step_call(self, var_vector):
        print(self.count)
        self.count += 1
        filename = '%s_img.jpg' % (self.count)
        save = skimage.transform.resize(var_vector,(300,300))
        skimage.io.imsave("./images/" + filename, save)
