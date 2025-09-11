import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load your trained multi-task CNN once
MODEL_PATH = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\cnn_multi_task_model.h5"
cnn_model = load_model(MODEL_PATH)

# Define possible classes
GENDER_CLASSES = ["Female", "Male"]
HANDEDNESS_CLASSES = ["Left", "Right"]
AGE_CLASSES = ["Child", "Teen", "Adult", "Senior"]  # adapt to your labels
STYLE_CLASSES = ["Cursive", "Print", "Mixed"]       # adapt to your labels

def preprocess_image(image_bytes, target_size=(64, 64)):
    """Convert image bytes to normalized array for CNN."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_handwriting_features_cnn(image_bytes):
    """Return predictions for Gender, Handedness, Age, and Writing Style."""
    img_array = preprocess_image(image_bytes)
    
    # CNN model output: [gender, handedness, age, style]
    preds = cnn_model.predict(img_array)
    
    # Convert predictions to readable form
    gender_pred = GENDER_CLASSES[int(preds[0][0] > 0.5)]
    gender_conf = float(preds[0][0] if gender_pred == "Male" else 1 - preds[0][0])
    
    handedness_pred = HANDEDNESS_CLASSES[int(preds[1][0] > 0.5)]
    handedness_conf = float(preds[1][0] if handedness_pred == "Right" else 1 - preds[1][0])
    
    age_pred = AGE_CLASSES[np.argmax(preds[2])]
    style_pred = STYLE_CLASSES[np.argmax(preds[3])]
    
    return {
        "gender": gender_pred,
        "gender_confidence": round(gender_conf * 100, 2),
        "handedness": handedness_pred,
        "handedness_confidence": round(handedness_conf * 100, 2),
        "age_group": age_pred,
        "writing_style": style_pred
    }

# Example usage
if __name__ == "__main__":
    with open(r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\female\F2.jpg", "rb") as f:
        img_bytes = f.read()
    
    predictions = predict_handwriting_features_cnn(img_bytes)
    print(predictions)
