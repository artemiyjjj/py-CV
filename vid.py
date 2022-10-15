import cv2
import numpy as np
import math

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,1000)
    cap.set(4,1200)

    cv2.namedWindow('Frame')
    cv2.createTrackbar("L-H", "Frame", 51 ,180, nothing)
    cv2.createTrackbar("L-S", "Frame", 0 ,255, nothing)
    cv2.createTrackbar("L-V", "Frame", 19 ,255, nothing)
    cv2.createTrackbar("U-H", "Frame", 108 ,180, nothing)
    cv2.createTrackbar("U-S", "Frame", 255 ,255, nothing)
    cv2.createTrackbar("U-V", "Frame", 156 ,255, nothing)

    while True:
        _, img = cap.read()

        l_h = cv2.getTrackbarPos("L-H", "Frame")
        l_s = cv2.getTrackbarPos("L-S", "Frame")
        l_v = cv2.getTrackbarPos("L-V", "Frame")
        u_h = cv2.getTrackbarPos("U-H", "Frame")
        u_s = cv2.getTrackbarPos("U-S", "Frame")
        u_v = cv2.getTrackbarPos("U-V", "Frame")
        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])

        blur_img = cv2.GaussianBlur(img, (7,7), 1)
        hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.erode(mask, kernel)
        cv2.imshow('mask', mask)
        cntr, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in cntr:

            rect = cv2.minAreaRect(cnt)  # вписываем прямоугольник
            box = cv2.boxPoints(rect)  # ищем четыре вершины прямоугольника
            box = np.int0(box)  # округляем координаты
            M = cv2.moments(cnt)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = [cX, cY]
            except ZeroDivisionError:
                center = (int(rect[0][0]), int(rect[0][1]))  # ищем центр прямоугольника
            area = int(rect[1][0] * rect[1][1])  # вычисляем площадь
            # вычисляем координаты двух векторов, являющихся сторонам прямоугольника
            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

            # выясняем какой вектор больше
            usedEdge = edge1
            if cv2.norm(edge2) > cv2.norm(edge1):
                usedEdge = edge2
            reference = (1, 0)  # горизонтальный вектор, задающий горизонт

            # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
            angle = 180.0 / math.pi * math.acos(
                (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge)))
            # отсекаем лишние контуры
            if area > 3000 and area < 30000:
                shape = ""
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                if len(approx) == 3:
                    shape = "triangle"

                elif len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)

                    shape = "rectangle"

                else:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    shape = "oval"
                cv2.drawContours(img, cnt, -1, (45, 235, 67), 2)
                cv2.circle(img, center, 5, (230, 0, 200), 2)  # рисуем маленький круг в центре фигуры
                # выводим на изображение величину угла наклона
                if shape!="oval":
                    cv2.putText(img, "%d" % int(angle), (center[0] + 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 235, 250), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(img, shape, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 1)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
main()

