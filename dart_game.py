import cv2
from ultralytics import YOLO
import calibration as cl

model_darts = YOLO("models/darts_model_improved.pt")
image = cv2.imread("debug/dart_picture.jpg")

is_calibrated, target = cl.calibrate(image)

results = model_darts(target)
size_image = image.shape
dim = (int(size_image[0]*0.25),int(size_image[1]*0.25))

image = results[0].plot()
image = cv2.resize(image,dim)

cv2.imshow("darts",image)
cv2.waitKey(0)


cv2.destroyAllWindows()