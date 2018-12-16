import tensorflow as tf
import numpy as np
import vgg19
import copy
import scipy
import skimage
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

VGG_MEAN = [103.939, 116.779, 123.68]

class neural_style(object):
    def __init__(self,
                content_layers = "conv5_2",
                style_layers = "conv1_1,conv2_1,conv3_1,conv4_1,conv5_1",
                content_img = None,
                style_img = None,
                initializer = "content"):
        #assertion
        assert content_img is not None and style_img is not None
        assert content_img.shape == style_img.shape
        
        #read input images
        self.vgg = vgg19.Vgg19()
        self.content_img = np.float32(content_img)
        self.style_img = np.float32(style_img)
        
        #initializing output image and noise image performs bad so we discard it
        if initializer == "style":
            init = tf.constant_initializer(np.array(self.style.img))
        elif initializer == "content":
            init = tf.constant_initializer(np.array(self.content_img))
        self.output_shape = self.content_img.shape
        shape = self.output_shape
        with tf.variable_scope("generated") as scope:
            self.output = tf.get_variable(name='output_image', shape = [1,shape[1],shape[2],3],
                                        dtype = tf.float32, initializer=init)
        
        #build vgg19 model
        self.vgg.build(self.output)
        self.content_layers = content_layers.split(',')     
        self.style_layers = style_layers.split(',')
        print("Content feature layers:%s" % (self.content_layers))
        print("Style feature layers:%s" % (self.style_layers))
        self.layers = {'conv1_1' : self.vgg.conv1_1,
                          'conv1_2' : self.vgg.conv1_2,
                          'conv2_1' : self.vgg.conv2_1,
                          'conv2_2' : self.vgg.conv2_2,
                          'conv3_1' : self.vgg.conv3_1,
                          'conv3_2' : self.vgg.conv3_2,
                          'conv3_3' : self.vgg.conv3_3,
                          'conv3_4' : self.vgg.conv3_4,
                          'conv4_1' : self.vgg.conv4_1,
                          'conv4_2' : self.vgg.conv4_2,
                          'conv4_3' : self.vgg.conv4_3,
                          'conv4_4' : self.vgg.conv4_4,
                          'conv5_1' : self.vgg.conv5_1,
                          'conv5_2' : self.vgg.conv5_2,
                          'conv5_3' : self.vgg.conv5_3,
                          'conv5_4' : self.vgg.conv5_4}
        
        #pre computation
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.c_layers = [sess.run(self.layers[i], {self.output : self.content_img}) for i in self.content_layers]
            self.s_layers = [sess.run(self.layers[i], {self.output : self.style_img}) for i in self.style_layers]

    def run(self,
            alpha = 0.01,
            beta = 1,
            gamma = 0.001,
            style_weights = [0.2,0.2,0.2,0.2,0.2],
            optimizer = "l-bfgs-b",
            learning_rate = 2,
            iterations = 500):
        self.style_weights = style_weights
        self.count = 0
        shape = self.output_shape
        self.loss = []
        
        start_time = time.time()
        
        with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
            
            #get content loss as a list for each layer
            content_loss = self.get_cont_loss()
            
            #get style loss as a list for each layer
            style_loss = self.get_style_loss()
            
            #total variance loss to suppress image noise
            tv_loss = tf.reduce_sum(tf.image.total_variation(self.output))
            
            total_loss = alpha * tf.reduce_sum(content_loss)+ beta * tf.reduce_sum(style_loss) + gamma * tv_loss
            
            if optimizer == "l-bfgs-b":
            #using l-bfgs-b optimizing algorithm
                sess.run(tf.global_variables_initializer())
                step = tf.contrib.opt.ScipyOptimizerInterface(total_loss,
                                                      method = "L-BFGS-B",
                                                      options = {"maxiter": iterations,"disp": 100})
                step.minimize(sess, loss_callback = self.loss_callback_func, fetches = [total_loss, self.output])
            elif optimizer == "Adam":
                step = tf.train.AdamOptimizer(learning_rate).minimize(loss = total_loss)
            elif optimizer == "Adadelta":
                step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss = total_loss)
            elif optimizer == "Adagrad":
                step = tf.train.AdagradOptimizer(learning_rate).minimize(loss = total_loss)
            
            if optimizer != "l-bfgs-b":
                sess.run(tf.global_variables_initializer())
                for i in range (iterations):   
                    current_loss = sess.run(total_loss)
                    sess.run(step)
                    self.loss.append(current_loss)
                    if i % 100 == 0:
                        print("loss at {}th iterarion:{}".format(i,current_loss))
                        next_output = sess.run(self.output)
                        next_output = next_output.reshape(shape[1], shape[2], 3)
                        imgplot = plt.imshow(next_output)
                        plt.axis('off')
                        plt.show()
          
            next_output = sess.run(self.output)
        print(("Stylizing time: %ds" % (time.time() - start_time)))
        return self.loss, next_output

    def get_cont_loss(self):
        losses = []
        P = self.c_layers
        F = [self.layers[i] for i in self.content_layers]
        for i in range(len(P)):
            shape = P[i].shape
            M,N = shape[1] * shape[2], shape[3]
            M = tf.convert_to_tensor(M, dtype = tf.int32)
            N = tf.convert_to_tensor(N, dtype = tf.int32)
            M = tf.to_float(M)
            N = tf.to_float(N)
            loss = tf.reduce_sum(tf.squared_difference(F[i], P[i]))
            loss /= 2
            losses.append(loss)
        return losses

    def get_style_loss(self):
        losses = []
        style_weights = self.style_weights
        feature_s = self.s_layers
        feature_o = [self.layers[i] for i in self.style_layers]
        for i in range(len(feature_s)):
            shape = feature_s[i].shape
            M,N = shape[1]*shape[2], shape[3]
            vect_s = tf.reshape(feature_s[i],[M,N])
            vect_o = tf.reshape(feature_o[i],[M,N])
            A = tf.matmul(vect_s, vect_s, transpose_a = True)
            G = tf.matmul(vect_o, vect_o, transpose_a = True)
            M=tf.convert_to_tensor(M, dtype=tf.int32)
            N=tf.convert_to_tensor(N, dtype=tf.int32)
            M=tf.to_float(M)
            N=tf.to_float(N)
            loss = tf.reduce_sum(tf.squared_difference(A, G))
            loss/=4*tf.multiply(N,N)*tf.multiply(M,M)
            loss = tf.multiply(style_weights[i], loss)
            losses.append(loss)
        return losses

    def loss_callback_func(self, loss, image):
        self.count += 1
        self.loss.append(loss)
        if self.count%100 == 0:
            print("Iteration %d: loss = %s" % (self.count,loss))
            shape = self.output_shape
            next_output = image
            next_output = next_output.reshape(shape[1], shape[2], 3)
            imgplot = plt.imshow(next_output)
            plt.axis('off')
            plt.show()

