# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import sys
import numpy as np
import img_processing
from tensorflow.keras.models import load_model

from PyQt5.QtGui import QPixmap

MainUI = "./main_test.ui"
app = QApplication(sys.argv)

class MainWindow(QMainWindow):
    def __init__(self) -> object:
        super().__init__()
        uic.loadUi(MainUI, self)

        #초기값 설정
        self.image = None #원본 이미지
        self.roi = None #ROI 이미지
        self.output = None #골연령 예측 결과


        self.gender = np.array([0]) #성별. 화면에서 Male 체크를 기본으로 함. 입력변수 Female [0], Male [1]
        self.model = load_model('./models/checkpoint-50-epochs-16-batchs.h5')

        #초기 화면에서 sample 이미지 보여주기
        #QPixmap이란, PyQt에서 이미지를 보여줄 때 사용하는 객체로 위에 사진에 있는 포맷들을 지원하는 객체
        self.qPixmapFile_origin = QPixmap()
        #샘플이미지 설정, 경로설정
        self.qPixmapFile_origin.load("./image/sample_origin.png")
        #사이즈 설정, 사이즈는 PYQT디자이너에서 정한 사이즈 내에서 새로 정할 수 잇음.
        self.qPixmapFile_origin = self.qPixmapFile_origin.scaled(450, 500)
        # 위에서 설정한 QPixmap을 표시하는 메서드, 마치 마지막 마침표와 같아 없어면 포시 X
        self.label_origin.setPixmap(self.qPixmapFile_origin)



        #버튼 클릭시 함수 연결
        self.pushButton_upload.clicked.connect(self.openFileNameDialog) #Upload
        self.radioButton_male.clicked.connect(self.gender_checked) #Male
        self.radioButton_female.clicked.connect(self.gender_checked) #Female



    def openFileNameDialog(self):  # opencv로 이미지 불러와서 qimage resize로 사이즈 줄인 후 label에 보여주기.
        fileName, _ = QFileDialog.getOpenFileName(self, "파일 선택", "",
                                                  "Image files (*.jpg *.gif *.png)")

        if fileName:
            self.setWindowTitle(fileName)
            self.fileName = fileName
            self.image = cv2.imdecode(np.fromfile(fileName, np.uint8), cv2.IMREAD_COLOR)  # cv2.imread()

            self.label_origin_show(self.image)
            self.wrist = img_processing.img_roi(self.image)
            self.roi = img_processing.print_roi(self.image)
            self.label_roi_show(self.wrist)

            self.output = self.bone_age_pred(self.roi)
            self.label_prediction_show(self.output)

    #좌측 화면(label_origin)에 이미지 보여주는 함수
    def label_origin_show(self, image):
        self.img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1]*3,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
        self.img = QtGui.QImage(self.img).scaled(450, 500, QtCore.Qt.KeepAspectRatio)  # GUI에 보여주기 위한 용도로 사진 줄이기
        self.label_origin.setPixmap(QtGui.QPixmap.fromImage(self.img))


    #아래 좌측 화면(label_roi)에 이미지 보여주는 함수
    def label_roi_show(self, image):
        #print(image.shape)
        self.img = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
        self.img = QtGui.QImage(self.img).scaled(180, 180, QtCore.Qt.KeepAspectRatio)
        self.label_roi.setPixmap(QtGui.QPixmap.fromImage(self.img))



    #Male, Female 체크시 성별 변경
    def gender_checked(self):
        if self.radioButton_male.isChecked():
            self.gender = np.array([1])
        if self.radioButton_female.isChecked():
            self.gender = np.array([0])

        if self.wrist is not None: #이미지 업로드 후 성별 변경시 골연령 다시 예측
            if type(self.gender) == type(self.roi): #이미지 업로드 후 roi가 array 타입인 경우, 성별 체크시 골연령 다시 예측
                self.output = self.bone_age_pred(self.roi)
                self.label_prediction_show(self.output)

    #골연령 예측 모형
    def bone_age_pred(self, roi): #gender, roi 모두 array 타입
        input = roi
        model = load_model('./models/tjnet_model.h5')
        prediction = self.model.predict(input.reshape(-1, 256, 256, 5))
        output = str(round(prediction[0][0], 1))
        return output

    #골연령 예측 결과 보여주기(Main)
    def label_prediction_show(self, output):
        self.label_prediction.setText("Bone-Age : " + output + " years")

if __name__ == "__main__":
    main_window = MainWindow()
    main_window.show()
    app.exec_()
