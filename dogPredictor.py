import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def make_prediction(file):

  IMG_SIZE = 224
  # define the batch size 32
  BATCH_SIZE = 32

  labels_csv = pd.read_csv('F:\python programms\Dog Vision\labels.csv')
  labels = labels_csv['breed']
  labels = np.array(labels)
  unique_breeds = np.unique(labels)
  boolean_labels = [label == unique_breeds for label in labels]

  # function for preprocessing image
  def process_image(image_path, img_size = IMG_SIZE):
    """
    Takes images file path and turns image into a Tensor
    """
    # read image file
    image = tf.io.read_file(image_path)
    # turn image into tensor with 3 color channels RGB
    image = tf.image.decode_jpeg(image, channels=3)
    # convert the color channel values to from 0 to 255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


  # turn predictions probabilities into their respective label
  def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


  # create a function to turn data into batches
  def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    create batches of data out of image (x) and label (y) pairs.
    shuffles the data if it's training data but doesn't shuffle if it's validation data
    test data as input also (no labels)
    """
    # data is a test dataset, no labels
    if test_data:
      print('creating test data batches ...')
      data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths no labels
      data_batch = data.map(process_image).batch(BATCH_SIZE)
      return data_batch

  # create a function to load a trained model
  def load_model(model_path):
    """
    loads a saved model from a specified path
    """
    print(f'loading saved model from: {model_path}')
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

  # load the trained model
  loaded_all_image_model = load_model("F:\python programms\Dog Vision\mymodel.h5")

  # get custom image
  custom_image_paths = [file]

  # turn custom image into batch dataset
  custom_data = create_data_batches(custom_image_paths, test_data=True)
  custom_data

  # make predictions 
  custom_preds = loaded_all_image_model.predict(custom_data)


  # get custom image prediction labels
  custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
  print(custom_pred_labels)

  return custom_pred_labels
