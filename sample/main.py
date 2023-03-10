######################################################
# PyCamera.py                                        #
# Copyright (c) 2021 By LiuHui. All Rights Reserved. #
#                                                    #
# Email: specterlh@163.com                           #
######################################################

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QCoreApplication, QThread, QObject, pyqtSignal
import cv2
import sys
import math
import numpy as np
import time
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

from ultralytics import YOLO

from typing import List, Mapping, Optional, Tuple, Union
from pyinstrument import Profiler

modeFlag = 0
model = YOLO("yolov8n.pt")

'''PyQt5窗口页面'''

class Ui_MainWindow(QtWidgets.QWidget):
    '''窗口初始化'''

    def __init__(self, window):
        super().__init__(window)
        self.MainWindow = window  # 定义窗口控件
        self.threads = []  # 定义多线程数组
        self.caps = []  # 定义摄像头数组
        self.labels = []  # 定义摄像头显示位置
        self.timers = []  # 定义定时器， 用于控制显示帧率
        self.setupUi()  # 定义窗口页面布局

    '''设置页面布局'''

    def setupUi(self):

        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(1560, 543)
        self.centralwidget = QtWidgets.QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.VideoBoxLabel = QtWidgets.QLabel(self.centralwidget)
        self.VideoBoxLabel.setGeometry(QtCore.QRect(10, 10, 640, 480))
        font = QtGui.QFont()
        font.setPointSize(35)
        font.setBold(True)
        font.setWeight(75)
        self.VideoBoxLabel.setFont(font)
        self.VideoBoxLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.VideoBoxLabel.setObjectName("VideoBoxLabel")
        self.TargetBoxLabel = QtWidgets.QLabel(self.centralwidget)
        self.TargetBoxLabel.setGeometry(QtCore.QRect(670, 10, 480, 480))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.TargetBoxLabel.setFont(font)
        self.TargetBoxLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.TargetBoxLabel.setObjectName("TargetBoxLabel")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(650, 20, 20, 461))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.line.setFont(font)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setGeometry(QtCore.QRect(1160, 10, 391, 161))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.StartButton.setFont(font)
        self.StartButton.setObjectName("StartButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1160, 420, 391, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1160, 260, 391, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1160, 340, 391, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.VisionButton = QtWidgets.QPushButton(self.centralwidget)
        self.VisionButton.setGeometry(QtCore.QRect(1160, 180, 391, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.VisionButton.setFont(font)
        self.VisionButton.setObjectName("VisionButton")
        self.MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1555, 28))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(self.MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(self.MainWindow)
        self.action_2.setObjectName("action_2")
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    '''设置控件显示文本'''

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.VideoBoxLabel.setText(_translate("MainWindow", "NO VIDEO SIGNAL"))
        self.TargetBoxLabel.setText(_translate("MainWindow", "NO TARGET"))

        self.StartButton.setText(_translate("MainWindow", "启动/暂停"))
        self.StartButton.clicked.connect(self.refreshCamera)

        self.pushButton_2.setText(_translate("MainWindow", "手势操控"))

        self.pushButton_3.setText(_translate("MainWindow", "手势判断"))
        self.pushButton_3.clicked.connect(self.switchGestureJudgment)

        self.pushButton_4.setText(_translate("MainWindow", "实时物品/图片识别"))
        self.pushButton_4.clicked.connect(self.switchYoloIdentify)

        self.VisionButton.setText(_translate("MainWindow", "可视化"))
        self.VisionButton.clicked.connect(self.switchHandsIdentify)

        self.menu.setTitle(_translate("MainWindow", "设置"))
        self.action.setText(_translate("MainWindow", "设置保存路径"))
        self.action_2.setText(_translate("MainWindow", "读取配置文件"))

    '''扫码已有摄像头，返回编号'''

    def GetUsbCameraIO(self):
        ids = []
        for i in range(10):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                ids.append(i)
        return ids

    '''关闭窗口'''

    def close_Window(self):
        for thread in self.threads:
            thread.quit()
        self.MainWindow.close()

    '''刷新摄像头布局'''

    def switchHandsIdentify(self):
        global modeFlag
        if modeFlag != 1:
            modeFlag = 1
        elif modeFlag == 1:
            modeFlag = 0

    def switchGestureJudgment(self):
        global modeFlag
        if modeFlag != 2:
            modeFlag = 2
        elif modeFlag == 2:
            modeFlag = 0

    def switchYoloIdentify(self):
        global modeFlag
        if modeFlag != 3:
            modeFlag = 3
        elif modeFlag == 3:
            modeFlag = 0

    def refreshCamera(self):
        # 停止现有的多线程
        for thread in self.threads:
            thread.quit()
        self.ids = self.GetUsbCameraIO()  # 获取当前所有可连接摄像头的id

        ids = np.array(self.ids)  # 将数组转换为矩阵
        n = len(ids)  # 获取矩阵长度
        cols = math.ceil(math.sqrt(n))  # 取矩阵开方下的最接近整数，向下取整
        rows = math.floor(math.sqrt(n))  # 取矩阵开方下的最接近整数，向上取整
        if cols * rows < n:  # 如果数量不够，增加一行
            rows = rows + 1
        arr = np.empty(cols * rows)  # 创建一个空矩阵
        arr[:n] = ids  # 空矩阵的前面为id矩阵
        arr[n:] = -1  # 空矩阵后面为-1
        arr = arr.reshape((rows, cols))  # 将矩阵转换为rows*cols的矩阵
        self.arr = arr
        # 循环创建布局及控件
        for i in range(rows):
            for j in range(cols):
                label = self.VideoBoxLabel
                label2 = self.TargetBoxLabel
                #self.gridLayout.addWidget(label, i, j, 1, 1)  # 设置label布局
                thread_i = QThread()  # 创建多个线程
                timer = QtCore.QTimer()  # 创建定时器
                cap = myCamera(label, label2, int(arr[i][j]), timer)  # 定义一个新的摄像机
                cap.signal.connect(self.flush)  # 摄像机可以触发状态改变
                cap.moveToThread(thread_i)  # 将摄像机绑定到线程
                timer.moveToThread(thread_i)  # 将定时器绑定到线程
                thread_i.started.connect(cap.started)  # 绑定线程起始事件
                thread_i.finished.connect(cap.finished)  # 绑定线程终止事件
                thread_i.start()  # 线程启动
                self.threads.append(thread_i)  # 记录线程
                self.caps.append(cap)  # 记录摄像机
                self.labels.append(label)  # 记录文本控件
                self.labels.append(label2) # 记录文本控件
                self.timers.append(timer)  # 记录计时器

    '''设置摄像头文本显示'''

    def flush(self, label, txt):
        label.setText(txt)


'''定义摄像机类'''


class myCamera(QObject):
    signal = pyqtSignal(QtWidgets.QLabel, str)  # 摄像机类返回信号

    '''初始化摄像机'''

    def __init__(self, label, label2, capIndex, timer):
        super(myCamera, self).__init__()
        self.label = label
        self.label2 = label2
        self.timer_camera = timer  # 帧数定时器
        self.capIndex = capIndex  # 相机编号

    '''设置显示图片'''

    def show(self):
        global modeFlag
        #pos = self.label.geometry()  # 获取label大小
        flag, image = self.cap.read()  # 读取摄像机图片
        image = cv2.flip(image,1)

        if modeFlag == 1 or modeFlag == 2 or modeFlag == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            """
            if results.multi_handedness:
                for hand_label in results.multi_handedness:
                    #print(results.multi_handedness.index)
                    print(hand_label)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    print('hand_landmarks:', hand_landmarks)
            """

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and modeFlag == 1:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if results.multi_handedness and (modeFlag == 2 or modeFlag == 3):  # 判断是否检测到手掌
                for i in range(len(results.multi_handedness)):
                    label = results.multi_handedness[i].classification[0].label  # 获得Label判断是哪几手
                    index = results.multi_handedness[i].classification[0].index  # 获取左右手的索引号
                    hand_landmarks = results.multi_hand_landmarks[i]

                    idx_to_coordinates = {}
                    point_u = 480
                    point_d = 0
                    point_l = 640
                    point_r = 0

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, 640, 480)
                        if landmark_px:
                            idx_to_coordinates[idx] = landmark_px
                            if (landmark_px[0]) > (point_r):
                                point_r = landmark_px[0]
                            if (landmark_px[0]) < (point_l):
                                point_l = landmark_px[0]
                            if (landmark_px[1]) > (point_d):
                                point_d = landmark_px[1]
                            if (landmark_px[1]) < (point_u):
                                point_u = landmark_px[1]

                    if len(idx_to_coordinates)==21:
                        angle_list = hand_angle(idx_to_coordinates)
                        gesture_str = h_gesture(angle_list)
                        #cv2.putText(image, gesture_str, (0, 100), 0, 1.3, (0, 0, 255), 3)

                    if (point_u != 640 and point_d != 0 and point_l != 640 and point_r != 0) and modeFlag == 2:
                        if index == 0:
                            cv2.rectangle(image, (point_l, point_u), (point_r, point_d), (80,127,255), 2)
                            if len(idx_to_coordinates) == 21:
                                cv2.putText(image, gesture_str, (point_l, point_d), 0, 1.3, (80,127,255), 2)
                        elif index == 1:
                            cv2.rectangle(image, (point_l, point_u), (point_r, point_d), (219,112,147), 2)
                            if len(idx_to_coordinates) == 21:
                                cv2.putText(image, gesture_str, (point_l, point_d), 0, 1.3, (219,112,147), 2)

                    if modeFlag == 3 and gesture_str == "one":
                        cv2.circle(image, idx_to_coordinates[8], 10, (80,127,255), 0)



        '''显示图片'''

        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色到rgb
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)  # 转换图片格式
        showImage = QtGui.QPixmap.fromImage(showImage)  # 转换图片格式
        self.label.setPixmap(showImage)  # 设置图片显示

        #self.label2.setPixmap(showImage)


    '''摄像头启动事件'''

    def started(self):
        self.timer_camera.timeout.connect(self.show)  # 设置计时器绑定
        self.signal.emit(self.label, "摄像头连接中，请稍候")  # 设置文本显示
        self.cap = cv2.VideoCapture(self.capIndex)
        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高度
        print("start:Camera" + str(self.capIndex))
        self.timer_camera.start(30)

    '''摄像头停止事件'''

    def finished(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label.setText("摄像头已断开")


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # fist five gun love one six three thumbup yeah
    '''
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None
    if 65535. not in angle_list:
        """
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbUp"
        """

        if (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "two"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle):
            gesture_str = "four"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"

    return gesture_str

def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow(mainWindow)
    ui.setupUi()
    mainWindow.show()
    sys.exit(app.exec_())



