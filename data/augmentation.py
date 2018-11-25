from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np 

# Dataset augmentation
seq = iaa.SomeOf((0, 4), [
    iaa.Noop(),
    iaa.Fliplr(0.5),  # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    iaa.GaussianBlur(sigma=(0, 3)),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.5, 1.5)),
    # Add gaussian noise.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.2),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.2, 2)),
    iaa.Sometimes(0.7,
                  # Apply affine transformations to each image.
                  # Scale/zoom them, translate/move them, rotate them and shear them.
                  iaa.Affine(
                      scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                      rotate=(-25, 25),
                      shear=(-8, 8)
                  )
                  ),
], random_order=True)  # apply augmenters in random order



def augmentation(features,labels, Multiply=2 ):
    """
    this function do augmentation on images an return it multiplied 
        :param features: 
        :param labels: 
        :param Multiply=2: 
    """
    old_labels = labels
    old_features = features

    for i in range(Multiply):
        features = np.concatenate(
            (features, seq.augment_images(old_features)), axis=0)
        labels = np.concatenate((labels, old_labels), axis=0)

    del old_labels, old_features
    return features, labels

