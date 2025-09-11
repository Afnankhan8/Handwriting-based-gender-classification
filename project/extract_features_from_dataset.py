import os
import numpy as np
from utils import preprocess_image, HandwritingFeatureExtractor

DATASET_DIR = r'C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset'
LABELS = {'male': 0, 'female': 1}  # folder names
features = []
labels = []

extractor = HandwritingFeatureExtractor()

for gender in LABELS:
    folder = os.path.join(DATASET_DIR, gender)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image = preprocess_image(path)
        feature_vector = extractor.extract_features(image)
        features.append(feature_vector)
        labels.append(LABELS[gender])

# Save features and labels
np.save('X.npy', np.array(features))
np.save('y.npy', np.array(labels))
print("✅ Features extracted and saved as X.npy and y.npy")
