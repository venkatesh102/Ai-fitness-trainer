import numpy as np
import cv2
import mediapipe as mp
import math
import time
import speech_recognition as sr
import pyttsx3 as p
speaker = p.init()

//Module 
class Module:
    def __init__(self, mode=False, model=1, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode

        self.model = model
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model, self.upBody,
                                     self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findposition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, landmark1,
                  landmark2, landmark3, draw=True):
        x1, y1 = self.lmList[landmark1][1:]
        x2, y2 = self.lmList[landmark2][1:]
        x3, y3 = self.lmList[landmark3][1:]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)

            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)

            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)

            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)

            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


detector = Module()
language = 'en'
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please say something....")
        audio = recognizer.listen(source, timeout=2)
        try:
            print("You said: \n" + recognizer.recognize_google(audio))
            return (recognizer.recognize_google(audio))
        except Exception as e:
            print("Error: " + str(e))

def text_to_speech(text):
    speaker.say(text)
    speaker.runAndWait()
    """ RATE"""
    rate = speaker.getProperty('rate')
    speaker.setProperty('rate', 150)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1280)
print('Ai......')
print("1.Right bicep curl\n2.Left bicep_curl\n3.Pushups\n4.Squats\n5.Skipping")
text_to_speech('Choose your exercise')
text_to_speech('press 1 for Right bicep curl')
text_to_speech("2 for left bicep curl")
text_to_speech('3 for push ups')
text_to_speech('4 for squats')
text_to_speech('5 for skipping')
text_to_speech('6 to end')
k = int(input("Press your option"))
#speech_to_text()
if k<6:
    text_to_speech(" Lets start ")
def right_bicep_curl():
    direction = 1
    count = 0
    m=-1
    pTime=0
    while True:
        success, img = cap.read
        img = detector.findPose(img, False)
        lmList = detector.findposition(img, False)

        if len(lmList) != 0:
            angle=detector.findAngle(img,12,14,16)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (210, 310), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)
            n = int(count)
            if n >= 1 and m is not n:
                text_to_speech(n)
                m = n
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)

            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670),cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (11, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("image", img)
        cv2.waitKey(1)
def left_bicep_curl():
    direction = 1
    count = 0
    m = -1
    pTime = 0
    while True:
        success, img = cap.read()
        # img = cv2.resize(img, (1024,800))
        img = detector.findPose(img, False)
        lmList = detector.findposition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (210, 310), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)
            n = int(count)
            if n >= 1 and m is not n:
                text_to_speech(n)
                m = n
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)

            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (11, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("image", img)
        cv2.waitKey(1)
def pushups():
    direction = 1
    count = 0
    m = -1
    pTime = 0
    while True:
        success, img = cap.read()
        img=cv2.resize(img,(1080,1080))
        img = detector.findPose(img, True)
        lmList = detector.findposition(img, False)

        if len(lmList) != 0:
            left_arm_angle=detector.findAngle(img,11,13,15)
            right_arm_angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(right_arm_angle, (140, 170), (0, 100))
            bar = np.interp(right_arm_angle, (140, 170), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)
            n = int(count)
            if n >= 1 and m is not n:
                text_to_speech(n)
                m = n
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)

            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (11, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("image", img)
        cv2.waitKey(1)
def squats():
    direction = 1
    count = 0
    m = -1
    pTime = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (720, 720))
        img = detector.findPose(img, False)
        lmList = detector.findposition(img, False)

        if len(lmList) != 0:
            right_angle = detector.findAngle(img,24, 26, 28)
            #left_angle=detector.findAngle(img,23,25,27)
            per = np.interp(right_angle, (190,240), (0, 100))
            bar = np.interp(right_angle, (190,240), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)
            n = int(count)
            if n >= 1 and m is not n:
                text_to_speech(n)
                m = n
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)

            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (11, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("image", img)
        cv2.waitKey(1)
def skipping():
    direction = 1
    count = 0
    m = -1
    pTime = 0
    while True:
        success, img = cap.read()
        # img = cv2.resize(img, (1024,800))
        img = detector.findPose(img, False)
        lmList = detector.findposition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 13, 15)
            angle1 = detector.findAngle(img, 12, 14, 16)
            angle2 = detector.findAngle(img, 24, 26,28)
            angle3 = detector.findAngle(img, 23, 25,27)
            per = np.interp(angle, (130, 145), (0, 100))
            bar = np.interp(angle, (130, 145), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if per == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            print(count)
            n = int(count)
            if n >= 1 and m is not n:
                text_to_speech(n)
                m = n
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)

            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (11, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.imshow("image", img)
        cv2.waitKey(1)
a=True
while(a):
    if k == 1:
        right_bicep_curl()
    elif k == 2:
        left_bicep_curl()
    elif k == 3:
        pushups()
    elif k == 4:
        squats()
    elif k==5:
        skipping()
    elif k==6:
        text_to_speech("Meet you again")
        a=False
    else:
        print("Enter correct option\n")
        text_to_speech("Enter the correct option")
