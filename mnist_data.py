import os
import struct
import numpy as np
 
def load_mnist(path, kind='train', abbrv=False):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' 
                                % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' 
                               % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
    # allow a smaller data set for algorithms unsuited to the larger
    if abbrv:
        if kind == 'train':
            return images[:6000], labels[:6000]
        elif kind == 'tk10':
            return images[:1000], labels[:1000]

    return images, labels
