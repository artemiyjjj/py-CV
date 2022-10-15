import cv2
import numpy as np
import math

img = cv2.imread('images/shapes.png')

def main():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,250, 250, 1)
    cntr, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in cntr:
        rect = cv2.minAreaRect(cnt)  # вписываем фигуру в прямоугольник
        box = cv2.boxPoints(rect)  # ищем четыре вершины прямоугольника
        box = np.int0(box)  # округляем координаты
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = [cX, cY]
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

        if area > 7000:
            shape = ""
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            if len(approx) == 3:
                shape = "triangle"

            elif len(approx) == 4:

                shape = "rectangle"

            elif len(approx) == 5:
                shape = "Pentagon"

            else:
                shape = "oval"

            cv2.drawContours(img, cnt, -1, (255, 0, 67), 3)
            cv2.circle(img, center, 5, (230, 0, 200), 2)  # рисуем маленький круг в центре фигуры
            # выводим вна первоначальное изображение величину угла наклона
            if shape != "oval":
                cv2.putText(img, "%d" % int(angle), (center[0] + 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 235, 250), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(img, shape, (x, y - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 1)
            f_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow('Result', f_img)
    cv2.waitKey(0)

main()

