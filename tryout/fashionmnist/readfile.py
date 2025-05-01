import tensorflow as tf

# Path to the TFRecord file
file_path = '/Users/druva/tensorflow_datasets/fashion_mnist/3.0.1/fashion_mnist-test.tfrecord-00000-of-00001'

# Define the feature description for parsing
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

# Function to parse each example
def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    })

    image = tf.io.decode_png(parsed['image'], channels=1)  # Try decode_jpeg if needed
    image = tf.image.resize(image, [28, 28])  # Optional if size is wrong
    image = tf.cast(image, tf.float32) / 255.0
    return image, parsed['label']
    # parsed = tf.io.parse_single_example(example_proto, feature_description)
    # image = tf.io.decode_raw(parsed['image'], tf.uint8)
    # image = tf.reshape(image, [28, 28])  # Fashion MNIST images are 28x28
    # image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    # label = parsed['label']
    # return image, label

# Load the dataset
dataset = tf.data.TFRecordDataset(file_path)
dataset = dataset.map(_parse_function)

# raw_dataset = tf.data.TFRecordDataset(file_path)
# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
#     print("Label:", label.numpy())

# Iterate through and display one example (optional)
for image, label in dataset.take(10):
    print("Label:", label.numpy())
    print("Image raw shape:", image.shape)
    import matplotlib.pyplot as plt
    plt.imshow(image.numpy(), cmap='gray')
    plt.title(f"Label: {label.numpy()}")
    plt.show()
