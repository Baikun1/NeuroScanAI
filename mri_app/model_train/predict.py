import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRIPredictor:
    """
    Class for predicting MRI tumor types using a pre-trained TensorFlow model.
    """

    def __init__(self, model_path=None):
        """
        Initialize the predictor by loading the model and setting class labels.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'mri_model.h5')
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Unable to load model from {model_path}: {e}")
        
        # Class labels (from the notebook)
        self.class_dict = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        self.classes = list(self.class_dict.keys())
        self.input_shape = (299, 299)  # Assuming InceptionV3 or similar

    def preprocess_image(self, img_path):
        """
        Preprocess the image: load, resize, convert to array, normalize.
        """
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure RGB
            resized_img = img.resize(self.input_shape)
            img_array = np.asarray(resized_img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {e}")
            raise ValueError(f"Invalid image file: {img_path}")

    def predict(self, img_path):
        """
        Predict the class of the MRI image.
        Returns: dict with predicted class and probabilities, or None on error.
        """
        if not os.path.exists(img_path):
            logger.error(f"Image path does not exist: {img_path}")
            return None
        
        try:
            img_array = self.preprocess_image(img_path)
            predictions = self.model.predict(img_array, verbose=0)
            probs = predictions[0]
            predicted_class = self.classes[np.argmax(probs)]

            result = {
                'predicted_class': predicted_class,
                'probabilities': {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            }
            logger.info(f"Prediction for {img_path}: {predicted_class}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed for {img_path}: {e}")
            return None
