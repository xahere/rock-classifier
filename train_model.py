import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# Hyperparameters & Config
DATA_DIR = Path("dataset_organized")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 10
EPOCHS_LITE = 5 # For quick retraining
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE = 1e-5
LEARNING_RATE_LITE = 1e-6 # Very slow learning rate for stability

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

def get_generators():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found.")

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
    
    return train_generator, validation_generator

def train_full():
    print(f"Initializing full training from {DATA_DIR}...")
    
    train_generator, validation_generator = get_generators()
    num_classes = len(train_generator.class_indices)
    
    # Compute balanced class weights
    train_classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_classes),
        y=train_classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    
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
    
    save_artifacts(model, train_generator.class_indices)


def retrain_lite():
    """
    Lightweight retraining function for the adaptive feedback loop.
    Loads existing model, trains for few epochs with low LR.
    """
    if not os.path.exists("rock_classifier.h5"):
        print("No existing model found for retraining. Running full training.")
        return train_full()
        
    print("Starting Lite Retraining (Adaptive Feedback)...")
    
    # Load generators (will include new feedback images now in the folders)
    train_generator, validation_generator = get_generators()
    
    # Load existing model
    model = load_model("rock_classifier.h5")
    
    # Re-compile with very low learning rate for stability
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_LITE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train for short duration
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS_LITE
    )
    
    save_artifacts(model, train_generator.class_indices)

def save_artifacts(model, class_indices):
    print("\nSaving model artifacts...")
    model.save("rock_classifier.h5")
    
    with open("class_indices.txt", "w") as f:
        for k, v in class_indices.items():
            f.write(f"{v}:{k}\n")
    print("Process completed successfully.")

if __name__ == "__main__":
    train_full()
