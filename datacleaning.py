from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

# Set up paths
train_dir = 'chest_xray/train'
valid_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

# Image preprocessing and augmentation (without shifting)
# Rescale pixel values to [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255,)  

valid_datagen = ImageDataGenerator(rescale=1./255) 

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=32,
    class_mode='binary' 
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)