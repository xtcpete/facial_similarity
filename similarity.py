import numpy as np
from PIL import Image
from matplotlib import pyplot
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import os
import cv2
from math import exp


# read the caffe model for face detection
face_detector = cv2.dnn.readNet("./build/deploy.prototxt", "./build/face_detect.caffemodel")


def img_process(filename):
    global face_detector

    #
    try:
        image = cv2.imread(filename)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        face_detector.setInput(blob)
        detections = face_detector.forward()
        count = 0
        (h, w) = image.shape[:2]
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.8:
                count += 1
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability

                y = startY - 10 if startY - 10 > 10 else startY + 10
                face_image = cv2.resize(image[startY: endY, startX:endX + 5], (224, 224))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            return face_image
    except:
        print(filename)
        pass


def extract_face(filename, required_size=(224,224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = Image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    faces = [img_process(f) for f in filenames]
    samples = np.asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    yhat = model.predict(samples)
    return yhat


def get_dist(known_embeddins, test_embeddings):
    dists = []
    for emb in known_embeddins:
        dist = np.sqrt(np.sum(np.square(test_embeddings - emb)))
        dists.append(dist)
    return dists


# read the database
def read_data(path):
    for r, d, f in os.walk(path):
        filenames = []
        for s in f:
            filenames.append(path + '/' + s)
    return filenames


def predict_result(test_file):
    test_embeddings = get_embeddings([test_file])[0]
    dists = get_dist(known_embeddins=embeddings, test_embeddings=test_embeddings)
    index = dists.index(min(dists))
    similarity = 1 / (1 + exp(0.0759 * dists[index] - 7.15))
    print("Similarity is：{:.2f}%".format(similarity * 100))

    file = filenames[index]
    similar_face = img_process(file)
    test_face = img_process(test_file)
    pyplot.rcParams['font.sans-serif'] = ['FangSong']
    pyplot.subplot(1, 2, 1)

    # pyplot.title(filenames[index])
    pyplot.imshow(similar_face)
    pyplot.axis('off')
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(test_face)
    pyplot.axis('off')
    pyplot.suptitle("Similarity is：{:.2f}%".format(similarity * 100))
    pyplot.show()


if __name__ == '__main__':

    path = "./data"
    # read the faces that we are comparing to
    filenames = read_data(path)

    embeddings = get_embeddings(filenames)

    stop = False
    while not stop:
        test_file = input("Please enter the filename, enter 0 to exit: ") # please put the file under test dictionary
        path = "./test/"+test_file
        if test_file != 0:
            try:
                predict_result(path)
            except:
                print("Please try again")
                continue
        else:
            exit()