# for for processing image data and building CNN
import tensorflow as tf
# give datasets to use
import tensorflow_datasets as tfds
#for displaying images and creating graphs
import matplotlib.pyplot as plt

# makes training results more reproducible
tf.keras.utils.set_random_seed(42)

# Load the cats vs dogs dataset and split into three groups:
# 70% training, 15% validation, 15% testing
(train_data, val_data, test_data), info = tfds.load(
    'cats_vs_dogs',
    split = ['train[:70%]', 'train[70%:85%]', 'train[85%:]'],

    # return each dataset example as a tuple (image, label)
    as_supervised = True,
    with_info = True
)

print("Dataset loaded successfully!")
print("Total images:", info.splits['train'].num_examples)


for image, label in train_data.take(1):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())
    print("Animal:", info.features['label'].int2str(label.numpy()))

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def preprocess(image, label):
    # resize each image to 64x64 so all images have the same dimensions
    image = tf.image.resize(image, (64, 64))

    # normalize pixel values from 0-255 to 0-1 for model training
    image = image / 255.0

    # return the modified image and original label(cat or dog)
    return image, label



for image, label in train_data.take(1):
    processed_image, processed_label = preprocess(image, label)

    print("Original shape:", image.shape)
    print("Processed shape:", processed_image.shape)

    print("Original type:", image.dtype)
    print("Processed type:", processed_image.dtype)


# apply preprocessing to each training image, shuffle each one
# then group them into batches of 32
train_data = train_data.map(preprocess).shuffle(1000).batch(32)


# no shuffling needed since validation doesn't update model weights
val_data = val_data.map(preprocess).batch(32)

test_data = test_data.map(preprocess).batch(32)

# inspect one batch to confirm that preprocessing worked
for images, labels in train_data.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)


# Build a CNN where each layer passes its output to the next
model = tf.keras.Sequential([
    # Specify the expected shape of one RGB input image
    tf.keras.Input(shape=(64, 64, 3)),

    # a Conv2D layer with 32 filters, 3x3 filter, ReLU activation, no padding
    tf.keras.layers.Conv2D(32, (3,3), activation = "relu", padding = "valid"),

    #gets new feature map of max values from each 2x2 region of input feature map
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

    # 2nd Conv2d layer for detecting more complex features
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", padding = "valid"),

    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

    # Turns the 14x14x64 feature maps into one 1D vector
    tf.keras.layers.Flatten(),

    # Combine features to classify the image
    tf.keras.layers.Dense(128, activation = "relu"),

    # Randomly set 50% of the input activations to zero during training
    tf.keras.layers.Dropout(0.5),

    # Final output neuron calculates score for cat(0) vs dog(1)
    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

# Display the layers and their output shapes
model.summary()


# sets up training
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)


# stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    restore_best_weights = True
)

# train for up to 10 epochs, stops early if improvement stops
history = model.fit(
    train_data,
    epochs = 10,
    validation_data = val_data,
    callbacks = [early_stopping],
    shuffle = False
)

# saves CNN and its learned weights
model.save("cat_dog_cnn.keras")

# Get epoch numbers starting from 1 instead of 0
epochs = range(1, len(history.history['accuracy']) + 1)

# Plot training and validation accuracy across epochs
plt.figure()
plt.plot(epochs, history.history['accuracy'], label = 'Training')
plt.plot(epochs, history.history['val_accuracy'], label = 'Validation')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training vs Validation Accuracy')
plt.xticks(epochs)
plt.legend()
plt.savefig("accuracy.png", dpi = 150, bbox_inches = "tight")
plt.show()


# Plot training and validation loss to identify possible overfitting
plt.figure()
plt.plot(epochs, history.history['loss'], label = 'Training')
plt.plot(epochs, history.history['val_loss'], label = 'Validation')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training vs Validation Loss')
plt.xticks(epochs)
plt.legend()
plt.savefig("loss.png", dpi=150, bbox_inches = "tight")
plt.show()


# evaluate the CNN model with unseenimages
test_loss, test_accuracy = model.evaluate(test_data)

# Display final test results
print(f"Final test accuracy: {test_accuracy *100:.2f}%")
print(f"Final test loss: {test_loss:.4f}")



# get one batch of test images and labels
for images, labels in test_data.take(1):

    # predict the dog score for the first 6 images
    predictions = model.predict(images[:6], verbose = 0)

    # create a figure showing 6 predictions
    plt.figure(figsize = (10, 7))

    for i in range(6):
        # get the sigmoid output for each image
        score = predictions[i][0]

        predicted_label = "dog" if score >= 0.5 else "cat"

        # convert the numerical label to animal label
        actual_label = info.features['label'].int2str(
            labels[i].numpy()
        )

        # display the image and the prediction
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Predicted: {predicted_label}\nActual: {actual_label}")
        plt.axis("off")

    # adjust spacing and save the final visualization
    plt.tight_layout()
    plt.savefig("predictions.png", dpi = 150, bbox_inches="tight")
    plt.show()
