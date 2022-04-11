import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # подключение к камере
mp_Hands = mp.solutions.hands # хотим распознать руки
hands = mp_Hands.Hands(max_num_hands = 2) # задаем хар-ки для распознавания
mpDraw = mp.solutions.drawing_utils # инцилизация для рисования рук или условия рук...

finger_coord = [(8,6), (12,10), (16,14), (20,18)] # координаты "суставов" четырех пальцев,кроме большого!
thumb_coord = (4,3) # координаты для большого палеца!

while cap.isOpened():
    success, image = cap.read()
    prevTime = time.time()
    if not success:
        print("не удалось получить кадр с web-камеры!")
        continue
    image = cv2.flip(image, 1) # зеркально отражаем изображение
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_image)
    multilandMarks = result.multi_hand_landmarks
    if multilandMarks:
        for idx, handLms in enumerate(multilandMarks):
            lbl = result.multi_handedness[idx].classification[0].label
            # print(lbl)
        upCount = 0 # счётчик пальцев
        for handLms in multilandMarks:
            mpDraw.draw_landmarks(image, handLms,  mp_Hands.HAND_CONNECTIONS)
            fingerslist = [] # список ключевых точек в пикселях
            for Im in handLms.landmark:
                h, w, c = image.shape
                cx, cy = int(Im.x * w), (Im.y * h)
                fingerslist.append((cx, cy))
            side = "left"
            if fingerslist[5][0] > fingerslist[17][0]:
                side = 'right'

            for coordinate in finger_coord:
                if fingerslist[coordinate[0]][1] < fingerslist[coordinate[1]][1]:
                    upCount += 1
            if side == "left":   
                if fingerslist[thumb_coord[0]][0] < fingerslist[thumb_coord[1]][0]:
                    upCount += 1
            else:
                if fingerslist[thumb_coord[0]][0] > fingerslist[thumb_coord[1]][0]:
                    upCount += 1

        cv2.putText(image, str(upCount), (50,150), cv2.FONT_HERSHEY_DUPLEX, 5, (0,200,50), 12)    
        print(upCount)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    cv2.putText(image, f"FPS: {int(fps)}",(400, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240,100,0), 1)    
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

cap.release()

