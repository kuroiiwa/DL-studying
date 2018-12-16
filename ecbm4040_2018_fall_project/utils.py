import skimage
import skimage.io
import skimage.transform
import numpy as np


#load image and crop it for resizing = [333,500,3]
def load_image(path):
    #skimage read image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    
    # crop image using shorter edge
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    
    #resize to [333,500]
    resized_img = skimage.transform.resize(img, (333, 500))
    return resized_img

def save_image(image, path, optimizer, alpha, iterations, lr):
    output_path = path + ("output_%s_%s_%s_%s.jpg" % (optimizer, alpha, iterations, lr))
    print("output image:%s" % (output_path))
    output = image.reshape(333,500,3)
    output /= 255
    skimage.io.imsave(output_path, output)