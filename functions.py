import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def load_images(path):
    images = []
    class_name = []
    my_list_image = os.listdir(path)
    for img in my_list_image:
        currentImg = cv2.imread(f'{path}/{img}')
        images.append(currentImg)
        class_name.append(os.path.splitext(img)[0])
    return images, class_name


def save_encode_images(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    np.save('encodes_images.npy', encode_list)


def load_encode_images(file):
    return np.load(file)


def register_recognition(name):
    with open('register.csv', 'r+') as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            data_current = datetime.now()
            data_current = data_current.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{data_current}')


#images, class_name = load_images('images')
#save_encode_images(images)