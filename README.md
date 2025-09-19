Handwriting-Based Gender Classification ✍️👩‍🧑

This project applies Machine Learning / Deep Learning techniques to classify gender based on handwriting samples. Handwriting contains unique biometric features, and by analyzing strokes, curves, and styles, we can build a model that predicts whether the writer is male or female.

Features

Preprocessing of handwriting images (grayscale, thresholding, normalization)

Feature extraction (classical + deep learning approaches)

Model training and evaluation using algorithms (CNN, SVM, etc.)

Accuracy reports, confusion matrix, and visualizations

Easy-to-extend for other handwriting-based biometric tasks

Project Structure
Handwriting-based-gender-classification/
│
├── dataset/                # Handwriting samples (train/test)
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Source code (models, preprocessing, training)
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── model.py
├── results/                # Trained models, metrics, and plots
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

Installation

Clone the repository

git clone git@github.com:Afnankhan8/Handwriting-based-gender-classification.git
cd Handwriting-based-gender-classification


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt

Usage
Training the model
python src/train.py --dataset dataset/ --epochs 20 --batch-size 32

Evaluating the model
python src/evaluate.py --model results/best_model.pth --test dataset/test

Results

Achieved XX% accuracy using CNN on the handwriting dataset

Compared with SVM and Random Forest models

Includes confusion matrix and prediction visualizations

(Update this section with your actual results after training)

Future Work

Extend dataset for better performance

Try transformer-based models

Deploy as a web app with Flask or Streamlit

Add classification for age group or handedness

Contributing

Contributions are welcome. Fork this repository, create a branch, and submit a pull request.

License

This project is licensed under the MIT License.
