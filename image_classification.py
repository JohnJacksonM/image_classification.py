import tensorflow as tf
from tensorflow import keras

class ImageClassifier:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.class_names = ["cat", "dog"]

    def preprocess_image(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, 0)
        return image

    def classify_image(self, image_path):
        image = self.preprocess_image(image_path)
        predictions = self.model.predict(image)
        predicted_class = self.class_names[tf.argmax(predictions, axis=1)[0]]
        return predicted_class

# Example usage:
model_path = "path/to/your/model.h5"
image_path = "path/to/your/image.jpg"

classifier = ImageClassifier(model_path)

predicted_class = classifier.classify_image(image_path)
print("Predicted class:", predicted_class)
