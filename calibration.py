import cv2
import numpy as np
from ultralytics import YOLO

model_dartboard = YOLO("models/dartboards_model_improved.pt")

image = cv2.imread("debug/dart_picture.jpg")
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

results = model_dartboard(image)
image_dessinee = results[0].plot()

size_image = image_dessinee.shape
dim = (int(size_image[0]*0.25),int(size_image[1]*0.25))
image_resized = cv2.resize(image_dessinee,dim)

image2 = image

mask = results[0].masks.xy #retourne les contours de la cible
mask_entiers = [c.astype(np.int32) for c in mask] # cv2 a besoin d'entiers 32bits, la ou YOLO donne des doubles.
cv2.drawContours(image, mask_entiers, -1, (0,255,0), 3)
image = cv2.resize(image,dim)

ellipse = cv2.fitEllipse(mask_entiers[0])
    
    # 2. Dessiner l'ellipse lissée sur l'image
    # On utilise une couleur différente (ex: Rouge) pour voir la différence
cv2.ellipse(image2, ellipse, (0, 0, 255), 3)
image2 = cv2.resize(image2,dim)

cv2.imshow("Test",image_resized)
cv2.waitKey(0)
cv2.imshow("Test",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()