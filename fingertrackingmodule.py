import cv2
import mediapipe as mp

class fingerDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectinCon=0.75, track=0.5):
        self.mpHands = mp.solutions.hands # хотим распознать руки
        self.hands = self.mpHands.Hands(mode, maxHands, complexity, detectinCon, track) # задаем хар-ки для распознавания
        self.mpDraw = mp.solutions.drawing_utils# инцилизация для рисования рук или условия рук...
        self.fingertips = [4,8,12,16,20]#кончики пальцев

    def findHands(self, img, draw=True):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGB_image)
        if draw:
            for handLms in self.result.multi_hand_Landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpDraw.HAND_CONNECTIONS)
        
        return img
