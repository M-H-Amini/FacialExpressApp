from inference import FaceExpressionRecognition
import torch
import cv2


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fer = FaceExpressionRecognition(model_adr="saved_model.pth", device="cpu")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ##  Detect faces using Haar Cascade Classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        ##  Draw rectangle around the faces and predict the emotion
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            class_label, certainty = fer.predict(roi, verbose=False)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_label} - {round(certainty, 2)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    