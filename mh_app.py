import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PIL.ImageQt import ImageQt
from mh_back import MHHandler, MHRecord
import numpy as np
import sys
import cv2
from inference import FaceExpressionRecognition
from datetime import datetime
import torch

print('Loading model...')
feelings = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('app.ui', self)
        self.setWindowTitle('FacialExpress')
        self.handler = MHHandler()
        self.Tabs.addTab(uic.loadUi('webcam_tab.ui'), 'Webcam')
        self.Tabs.addTab(uic.loadUi('summary_tab.ui'), 'Summary')
        self.initWebcamTab()
        self.initSummaryTab()
        self.current_time, self.current_feeling, self.current_confidence = None, None, None
        self.fer = FaceExpressionRecognition(model_adr="saved_model.pth", device="cpu")
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def predict(self, img):
        class_label, certainty = None, None
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ##  Detect faces using Haar Cascade Classifier
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        ##  Draw rectangle around the faces and predict the emotion
        if len(faces):
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
            x, y, w, h = faces
            roi = gray[y:y+h, x:x+w]
            class_label, certainty = self.fer.predict(roi, verbose=False)
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 0, 0), 2)
            cv2.putText(img, f'{class_label} - {round(certainty, 2)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2)
        return img, class_label, certainty
        

    def loadTable(self):
        self.Table.setRowCount(len(self.handler.records))
        self.Table.setColumnCount(3)
        self.Table.setHorizontalHeaderLabels(['Date', 'Feeling', 'Confidence'])
        self.Table.horizontalHeader().setSectionResizeMode(0, 1)
        for i, record in enumerate(self.handler.records):
            self.Table.setItem(i, 0, QTableWidgetItem(record['date']))
            self.Table.setItem(i, 1, QTableWidgetItem(record['feeling']))
            self.Table.setItem(i, 2, QTableWidgetItem(f'{record["confidence"]}'))


    def initSummaryTab(self):
        self.Table = self.Tabs.widget(1).Table
        self.loadTable()
        ##  Top Left Plot...
        self.PlotBoxT = self.Tabs.widget(1).PlotBoxT
        sns.set()
        self.figure_t = plt.figure()
        self.ax_t = self.figure_t.add_subplot(111)
        self.canvas_t = FigureCanvas(self.figure_t)
        layout = QVBoxLayout(self.PlotBoxT)
        layout.addWidget(self.canvas_t)
        ##  Buttom Plot...
        self.PlotBoxB = self.Tabs.widget(1).PlotBoxB
        self.figure_b = plt.figure()
        self.ax_b = self.figure_b.add_subplot(111)
        self.canvas_b = FigureCanvas(self.figure_b)
        layout = QVBoxLayout(self.PlotBoxB)
        layout.addWidget(self.canvas_b)
        ##  Update Plots...
        self.updatePlots()
    
    def updatePlots(self):
        counts = [(len([record for record in self.handler.records if record['feeling_index'] == i]), feelings[i]) for i in range(len(feelings))]
        counts = list(filter(lambda x: x[0], counts))
        counts, labels = [x[0] for x in counts], [x[1] for x in counts]
        self.ax_t.clear()
        self.ax_t.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, radius=1.4, textprops={'fontsize': 5})
        self.canvas_t.draw()

    def initWebcamTab(self):
        self.StartWebcamButton = self.Tabs.widget(0).StartWebcamButton
        self.StartWebcamButton.clicked.connect(self.startWebcam)
        ##
        self.StopWebcamButton = self.Tabs.widget(0).StopWebcamButton
        self.StopWebcamButton.clicked.connect(self.stopWebcam)
        self.StopWebcamButton.setEnabled(False)
        ##
        self.WebcamLabel = self.Tabs.widget(0).WebcamLabel
        self.WebcamLabel.setAlignment(Qt.AlignCenter)
        ## 
        self.ExpLabel = self.Tabs.widget(0).ExpLabel
        self.ExpLabel.setAlignment(Qt.AlignCenter)
        ##
        self.ManualButton = self.Tabs.widget(0).ManualButton
        self.ManualButton.clicked.connect(self.manualAdd)
        self.ManualButton.setEnabled(False)

    def manualAdd(self):
        record = MHRecord(self.current_time, self.current_feeling, feelings.index(self.current_feeling), self.current_confidence)
        self.handler.addRecord(record)
        self.handler.save()
        self.loadTable()
        self.updatePlots()


    def startWebcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam_thread = WebcamThread(self.capture, self)
        self.webcam_thread.start()
        self.StartWebcamButton.setEnabled(False)
        self.StopWebcamButton.setEnabled(True)
        self.ManualButton.setEnabled(True)

    def stopWebcam(self):
        self.capture.release()
        self.webcam_thread.quit()
        self.WebcamLabel.clear()
        self.ExpLabel.setText('')
        self.StartWebcamButton.setEnabled(True)
        self.StopWebcamButton.setEnabled(False)
        self.ManualButton.setEnabled(False)

class WebcamThread(QThread):
    def __init__(self, capture, parent=None):
        super(WebcamThread, self).__init__()
        self.capture = capture
        self.parent = parent
    
    def run(self):
        while True:
            ret, frame = self.capture.read() 
            if ret:
                frame, prediction, confidence = self.parent.predict(frame)
                img_qt = ImageQt(Image.fromarray(frame[:, :, ::-1]))
                self.parent.WebcamLabel.setPixmap(QPixmap.fromImage(img_qt))
                self.parent.current_time = datetime.now()
                self.parent.current_feeling = prediction if prediction else 'Unknown'
                self.parent.current_confidence = confidence if confidence else 0
                self.parent.ExpLabel.setText(f'{self.parent.current_feeling} ({self.parent.current_confidence:.2f})')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())