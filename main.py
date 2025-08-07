# Import necessary libraries
import os
import tensorflow as tf
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

# NOTE: Assuming utils, model, data_generator, and visualize modules are available
# These are custom modules that need to be in the same directory or on the Python path
from utils import dice_coefficient, iou_metric
from model import unet_model 
from data_generator import DataGenerator
from visualize import visualize_predictions


# -------------------------------
# 1. Verify GPU Availability
# -------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

# -------------------------------
# 2. Prepare the Data
# -------------------------------

# Define the data directory (adjust this path if necessary)
# Ensure this path is correct for your environment after downloading the data.
data_dir = '/root/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3/BraTS2020_training_data/content/data'

# Get list of .h5 files
try:
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
except FileNotFoundError:
    raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please check the path and ensure the dataset is available.")

# Ensure that there are .h5 files in the directory
if not h5_files:
    raise FileNotFoundError(f"No .h5 files found in the directory: {data_dir}")

# Split into training and validation sets
train_files, val_files = train_test_split(h5_files, test_size=0.2, random_state=42)

print(f"Total samples: {len(h5_files)}")
print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Parameters
batch_size = 8  # Adjust based on GPU memory
image_dim = (128, 128)  # Adjust based on your image dimensions

# -------------------------------
# 3. Instantiate Data Generators
# -------------------------------
training_generator = DataGenerator(train_files, data_dir, batch_size=batch_size, dim=image_dim, shuffle=True)
validation_generator = DataGenerator(val_files, data_dir, batch_size=batch_size, dim=image_dim, shuffle=False)

# -------------------------------
# 4. Build and Compile the Model
# -------------------------------
# The model is built from scratch here since we are training for the first time.
model = unet_model(input_size=(*image_dim, 4))  # Updated input size
model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics=['accuracy', dice_coefficient, iou_metric])
model.summary()

# -------------------------------
# 5. Define Callbacks
# -------------------------------
# Create a timestamp for the logs
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# TensorBoard callback for visualization
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# EarlyStopping callback with multiple metric monitoring
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    verbose=1, 
    restore_best_weights=True
)

# ModelCheckpoint callback to save the best model with '.keras' extension
model_checkpoint = ModelCheckpoint(
    'model-unet.best.keras',  # Changed from '.h5' to '.keras'
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

# Combine all callbacks
callbacks = [
    early_stopping,
    model_checkpoint,
    tensorboard_callback
]

# -------------------------------
# 6. Train the Model
# -------------------------------
epochs = 5  

try:
    results = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 # Set to 1 for progress bar, 2 for one line per epoch, 0 for silent
    )
except Exception as e:
    print(f"An error occurred during training: {e}")

# -------------------------------
# 7. Save the Final Model (Optional)
# -------------------------------
model.save('final_model_unet.keras')
print("Model training complete and saved as 'final_model_unet.keras'.")


# -------------------------------
# 8. Visualize Predictions (using the newly trained model)
# -------------------------------
print("Visualizing predictions on validation set...")
# The 'model' variable is already the newly trained model from the fit() call,
# so we can use it directly.
visualize_predictions(validation_generator, model, num_samples=11430)

