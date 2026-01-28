import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# Hyperparameters & Config
DATA_DIR = Path("dataset_organized")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 10
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE = 1e-5

def create_model(num_classes):

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=outputs), base_model

def train():
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        return

    print(f"Initializing training with data from {DATA_DIR}...")

    # Data Augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Classes detected: {train_generator.class_indices}")

    # Compute balanced class weights to handle dataset imbalance
    print("Computing class weights...")
    train_classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_classes),
        y=train_classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Weights: {class_weights_dict}")
    
    model, base_model = create_model(num_classes)
    
    # --- Phase 1: Train Top Layers ---
    print("\nStarting Phase 1: Training Classification Head...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights_dict
    )
    
    # --- Phase 2: Fine-Tuning ---
    print("\nStarting Phase 2: Fine-Tuning Base Model...")
    base_model.trainable = True
    
    # Fine-tune from layer 100
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    callbacks = [
        ModelCheckpoint("rock_classifier_best.keras", save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS_FINE,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )
    
    print("\nSaving final model artifacts...")
    model.save("rock_classifier.h5")
    
    with open("class_indices.txt", "w") as f:
        for k, v in train_generator.class_indices.items():
            f.write(f"{v}:{k}\n")
            
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    train()
