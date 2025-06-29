import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
train_dir = r"C:\Users\bharg\Downloads\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Waste_Classification_Dataset\train"
test_dir = r"C:\Users\bharg\Downloads\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Waste_Classification_Dataset\test"

# Image generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all VGG layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('vgg16.h5', save_best_only=True)

# Train the model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    callbacks=[early_stop, checkpoint],
    steps_per_epoch=len(train_data),
    validation_steps=len(test_data)
)

print("✅ Model training complete. Saved as 'vgg16.h5'")
