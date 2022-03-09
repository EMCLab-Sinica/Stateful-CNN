import tensorflow_datasets as tfds

# https://www.tensorflow.org/datasets/catalog/imagenet_resized
# https://tf.wiki/zh_hans/appendix/tfds.html
dataset = tfds.load('imagenet_resized/32x32', split=tfds.Split.TRAIN, as_supervised=True)
for images, labels in dataset:
    print(images.shape)
    print(labels.shape)
