# Musical Instrument Classification using Mel Spectrograms

A Jupyter notebook implementation of musical instrument classification using mel spectrograms and CNN. The project uses the IRMAS (Instrument Musical in Audio Signals) dataset to identify different musical instruments from audio samples.

## 📊 Project Overview

This notebook demonstrates how to:
1. Load and preprocess audio files from IRMAS dataset
2. Convert audio to mel spectrograms using librosa
3. Build and train a CNN model for instrument classification
4. Evaluate model performance and make predictions

## 🛠️ Setup and Dependencies

```python
# Required imports
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
```

## 📓 Notebook Structure

The notebook is organized into the following sections:

1. **Data Loading and Preprocessing**
   - Loading audio files using librosa
   - Generating mel spectrograms
   - Creating training/validation splits

2. **Feature Extraction**
  

3. **Model Architecture**
   ```python
   def create_improved_model(input_shape, num_classes):
    """Create an improved CNN model with residual connections."""
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First block
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Second block
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third block
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Fourth block
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        
        # Output layer
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model
   ```

4. **Training Loop**
   - Model training with loss visualization
   - Validation accuracy tracking
   - Learning rate adjustment

5. **Evaluation and Testing**
   - Model performance visualization
   - Confusion matrix
   - Audio sample testing

## 🎸 Supported Instruments

- Cello (cel)
- Clarinet (cla)
- Flute (flu)
- Acoustic Guitar (gac)
- Electric Guitar (gel)
- Organ (org)
- Piano (pia)
- Saxophone (sax)
- Trumpet (tru)
- Violin (vio)



## 🚀 Usage

1. Open the notebook in Jupyter Lab/Notebook
2. Update the dataset path to your IRMAS dataset location
3. Run all cells in sequence
4. Use the prediction cells to test new audio files

## 💾 Requirements

- Python 3.8+
- Jupyter Notebook/Lab
- librosa
- torch
- numpy
- matplotlib
- pandas


## 📝 Notes

- Ensure audio files are in WAV format
- Adjust batch size based on your GPU/CPU capabilities
- Experiment with different mel spectrogram parameters for optimal results
