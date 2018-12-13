import numpy as np
import tensorflow as tf
import vgg19

def get_cont_loss_layer(output, content, layer):
    content_layers = layer.split(',')
    losses = []
    for l in content_layers:
        vgg_l = data[l]
        loss = content_loss(output, content, vgg_l)
        losses.append(loss)
    return losses

def content_loss(output, content, layer):
    with tf.Session() as sess:
        F = sess.run(layer, {images: content})
        P = sess.run(layer, {images: output})
    shape = F.shape
    m,n = shape[1]*shape[2], shape[3]
    F = F.reshape(m,n)
    P = P.reshape(m,n)
    loss = 0.5 * tf.reduce_sum(tf.squared_difference(F, P))
    return loss

def get_style_loss_layer(output, style, layer):
    style_layers = layer.split(',')
    losses = []
    for l in style_layers:
        vgg_l = data[l]
        loss = style_loss(output, style, vgg_l)
        losses.append(loss)
    return losses

def gram_matrix(image, vgg_l):
    with tf.Session() as sess:
        conv1_1 = sess.run(vgg_l, {images: image})
    shape = conv1_1.shape
    m,n = shape[1]*shape[2], shape[3]
    vect_layer = conv1_1.reshape(m,n)
    gram = tf.matmul(vect_layer, vect_layer, transpose_a = True)
    return gram,m,n

def style_loss(output, style, vgg_l):
    A,m,n = gram_matrix(style, vgg_l)
    G,m,n = gram_matrix(output, vgg_l)
    loss = 1/(4*n^2*m^2) * tf.reduce_sum(tf.squared_difference(A, G))
    return loss
