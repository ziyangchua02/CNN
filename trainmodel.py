import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50 # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
from datacleaning import train_generator, valid_generator, test_generator 

# Build the model using ResNet50 as a base (pre-trained on ImageNet)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model (we don’t want to update its weights during training)
base_model.trainable = True

# Create the full model
model = Sequential([
    base_model,  # Add the pre-trained ResNet50 base model
    GlobalAveragePooling2D(),  # Pooling to reduce the output size from ResNet50
    Dense(1024, activation='relu'),  # Fully connected layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer for binary classification (Pneumonia vs. Normal)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',  # Binary classification loss function
              metrics=['accuracy'])  # Track accuracy during training

# Print the model summary to confirm the architecture
model.summary()

# Train the model
history = model.fit(
    train_generator,  # Generator for training data # pyright: ignore[reportUndefinedVariable]
    epochs=2,  # Number of training epochs
    validation_data=valid_generator  # Validation data for evaluating the model during training # pyright: ignore[reportUndefinedVariable]
)

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the trained model to a file
model.save('pneumonia_resnet_model.h5')
