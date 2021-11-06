import cv2
import numpy as np
import face_recognition

from functions import load_images, load_encode_images, save_encode_images

images, class_name = load_images('images')
# executar função para criar arquivo com encodes das imagens apenas na primeira vez que executar o app
save_encode_images(images)
encode_list_known = load_encode_images('encodes_images.npy')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_loc_curr_frame = face_recognition.face_locations(img_small)
    encode_curr_frame = face_recognition.face_encodings(img_small, face_loc_curr_frame)

    for encode_face, face_location in zip(encode_curr_frame, face_loc_curr_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
        # print(face_distance)
        matche_index = np.argmin(face_distance)

        if matches[matche_index]:
            name = class_name[matche_index].upper()
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento Facial", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
