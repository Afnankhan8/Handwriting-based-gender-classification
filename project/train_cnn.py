import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Clear previous session
K.clear_session()

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Dataset path
DATASET_PATH = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset"

# ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Example: Use a generator with multi-labels (assumes your dataset has subfolders or CSV labels)
# Here, we'll assume labels are in a CSV like: filename, gender, handedness, age_group, style
# You will need a custom generator for multi-output training.

def multi_output_generator(datagen, dataframe, directory, batch_size, target_size=(64,64)):
    while True:
        batch_data = []
        batch_gender = []
        batch_handedness = []
        batch_age = []
        batch_style = []

        # Load batch
        for i, row in dataframe.sample(batch_size).iterrows():
            img_path = os.path.join(directory, row['filename'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            batch_data.append(img)

            batch_gender.append(row['gender'])
            batch_handedness.append(row['handedness'])
            batch_age.append(row['age_group'])
            batch_style.append(row['style'])

        yield (tf.convert_to_tensor(batch_data),
               {
                   'gender_output': tf.convert_to_tensor(batch_gender),
                   'handedness_output': tf.convert_to_tensor(batch_handedness),
                   'age_output': tf.convert_to_tensor(batch_age),
                   'style_output': tf.convert_to_tensor(batch_style)
               })

# Build base CNN
inputs = Input(shape=(64,64,3))

x = Conv2D(16,(3,3),activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(32,(3,3),activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64,(3,3),activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Multi-output layers
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)           # binary
handedness_output = Dense(1, activation='sigmoid', name='handedness_output')(x)   # binary
age_output = Dense(3, activation='softmax', name='age_output')(x)                  # 3 classes
style_output = Dense(3, activation='softmax', name='style_output')(x)              # 3 classes

# Model
model = Model(inputs=inputs, outputs=[gender_output, handedness_output, age_output, style_output])

# Compile model
model.compile(
    optimizer='adam',
    loss={
        'gender_output':'binary_crossentropy',
        'handedness_output':'binary_crossentropy',
        'age_output':'categorical_crossentropy',
        'style_output':'categorical_crossentropy'
    },
    metrics=['accuracy']
)

# You need to create a dataframe with all labels and then use multi_output_generator for training
# history = model.fit(train_gen, validation_data=val_gen, epochs=15)

# Save model
MODEL_PATH = os.path.join("cnn_multi_task_model.h5")
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
