import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    x_train = image
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        temp = np.zeros((40, 40, 3))
        temp[4:36, 4:36] = image
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        ran = np.random.randint(9, size = 2)
        x_train = temp[ran[0]:ran[0]+32, ran[1]:ran[1]+32]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        ran_flip = np.random.random()
        if ran_flip >= 0.5:
            x_train = np.flip(x_train, axis = 1)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(x_train, axis = (0,1))
    std = np.std(x_train, axis = (0,1))
    x_train = (x_train - mean) / std
    ### YOUR CODE HERE

    return x_train