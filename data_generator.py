import tensorflow as tf
from tensorflow.keras.utils import Sequence
from skimage.transform import resize
import albumentations as A
import h5py
import numpy as np
import os

# -------------------------------
# Define the Data Generator
# -------------------------------
class DataGenerator(Sequence):
    def __init__(self, h5_files, data_dir, batch_size=16, dim=(128, 128), shuffle=True, augment=True):
        self.h5_files = h5_files
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.h5_files))
        self.on_epoch_end()
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), p=0.5),
        ])
        
    def __len__(self):
        return int(np.floor(len(self.h5_files) / self.batch_size))
        
    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        h5_batch = [self.h5_files[k] for k in batch_indexes]
        X, y = self.__data_generation(h5_batch)
        return X, y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, h5_batch):
        # Initialize arrays
        X = np.empty((self.batch_size, *self.dim, 4), dtype='float32')  
        y = np.empty((self.batch_size, *self.dim, 4), dtype='float32')  
            
        for i, h5_file in enumerate(h5_batch):
            file_path = os.path.join(self.data_dir, h5_file)
            with h5py.File(file_path, 'r') as hf:
                if 'image' in hf.keys() and 'mask' in hf.keys():
                    image = hf['image'][:]  # (H, W, 4)
                    mask = hf['mask'][:]    # (H, W) or (H, W, C)
                else:
                    raise KeyError(f"Unexpected keys in {h5_file}: {list(hf.keys())}")
                
                if mask.ndim > 2:
                    mask = np.mean(mask, axis=-1)  # Convert multi-channel mask to single-channel
                
                if image.shape[:2] != self.dim:
                    image = resize(image, (*self.dim, image.shape[2]), preserve_range=True, anti_aliasing=True)
                
                if mask.shape != self.dim:
                    mask = resize(mask, self.dim, preserve_range=True, order=0, anti_aliasing=False)
                
                # Normalize image
                image = image.astype('float32') / 255.0
                
                # Data augmentation
                if self.augment:
                    augmented = self.augmentation(image=image, mask=mask)
                    image, mask = augmented['image'], augmented['mask']
                
                # One-hot encode mask
                mask = tf.keras.utils.to_categorical(mask, num_classes=4)
                
                X[i] = image  # (128, 128, 4)
                y[i] = mask  # (128, 128, 4)
            
        return X, y
    