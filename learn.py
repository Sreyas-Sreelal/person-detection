from tensorflow.keras.utils import image_dataset_from_directory,img_to_array
from tensorflow.keras import layers, Sequential,losses,models
from tensorflow import expand_dims
from tensorflow.nn import softmax
import numpy
import pathlib
import cv2

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180

def process_data(path):
    print("Processing images, extracting faces..")
    for x in path.glob("*"):
        for pp in x.glob("*.jpg"):
            image = cv2.imread(str(pp))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            if len(faces)>0:
                count = 0
                for x, y, w, h in faces:
                    count +=1
                    if(w < 200  or h<200):
                        continue
                    image = image[y:y+h,x:x+w]
                    p = str(pp).split('\\')
                    try:
                        cv2.imwrite('processed\\'+p[1] + '\\'+str(count)+p[2],image)
                    except:
                        continue

def train_data(training_set):
    data_augmentation = Sequential([
            layers.RandomFlip("horizontal",input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3)),
            layers.RandomRotation(0.1),
    ])
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(training_set.class_names), name="outputs")
    ])
    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(training_set,epochs=15)
    return model

def predict(model,image):
    img_array = img_to_array(image)
    img_array = expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = softmax(predictions[0])
    return numpy.argmax(score),numpy.max(score) * 100.0

def init(dir):
    
    try:
        model = models.load_model('model')
        data_dir = pathlib.Path('processed')
        training_set = image_dataset_from_directory(
            data_dir,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        )
    except IOError:
        data_dir = pathlib.Path(dir)
        process_data(data_dir)
        input("Processed data waiting for user to do a manual check, press any key if it's done: ")
        data_dir = pathlib.Path(dir)
        training_set = image_dataset_from_directory(
            data_dir,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        model = train_data(training_set)
        model.save('model')
    return model,training_set.class_names
