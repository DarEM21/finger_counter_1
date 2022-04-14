import cv2
import mediapipe as mp

class handDetector():
    def __init__(self,mode=False, maxHands=2, modelComplexite=1, detection=0.5, track=0.5):
        self.mp_Hands = mp.solutions.hands # хотим распознать руки
        self.hands = self.mp_Hands.Hands(mode, maxHands, modelComplexite, detection, track) # задаем хар-ки для распознавания
        self.mpDraw = mp.solutions.drawing_utils# инцилизация для рисования рук или условия рук...

        pass

    def findHands(self, img):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGB_image)
        pass

    def findPositionPoints(self):
        pass

