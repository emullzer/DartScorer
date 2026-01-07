import cv2
import numpy as np
from ultralytics import YOLO

model_dartboard = YOLO("models/dartboards_model_improved.pt")


def calibrate(image):
    results = model_dartboard(image)
    if results == None :
        return False
    
    mask = results[0].masks.xy #retourne les contours de la cible
    mask_entiers = [c.astype(np.int32) for c in mask] # cv2 a besoin d'entiers 32bits, la ou YOLO donne des doubles.
    ellipse = cv2.fitEllipse(mask_entiers[0])

    (xc, yc), (d1, d2), angle_deg = ellipse
    angle_rad = np.deg2rad(angle_deg) # OpenCV donne des degrés, on passe en radians

    # On calcule les demi-axes (rayons)
    r1 = d1 / 2
    r2 = d2 / 2

    # Calcul des 4 points cardinaux de l'ellipse inclinée
    # On utilise le cosinus et le sinus pour projeter les rayons avec l'angle
    pts_src = np.array([
        [xc - r1 * np.cos(angle_rad), yc - r1 * np.sin(angle_rad)], # Gauche
        [xc + r1 * np.cos(angle_rad), yc + r1 * np.sin(angle_rad)], # Droite
        [xc + r2 * np.sin(angle_rad), yc - r2 * np.cos(angle_rad)], # Haut
        [xc - r2 * np.sin(angle_rad), yc + r2 * np.cos(angle_rad)]  # Bas
    ], dtype="float32")

    size_target = 600
    pts_dst = np.array([
        [0,size_target//2],#gauche
        [size_target,size_target//2],#droite
        [size_target//2,0],#haut
        [size_target//2,size_target]#bas
    ],dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src,pts_dst)
    squared_image = cv2.warpPerspective(image,M,(size_target,size_target))


    return True,squared_image

def show_contours(image,results):
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
    print(ellipse)

    image2 = cv2.resize(image2,dim)

    cv2.imshow("Test",image_resized)
    cv2.waitKey(0)
    cv2.imshow("Test",image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main():
    image = cv2.imread("debug/dart_picture.jpg")
    is_calibrated,image_result = calibrate(image)
    if is_calibrated : 
        cv2.imshow("Resized image",image_result)
        cv2.waitKey(0)
    else:
        print("fonctionne pas")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()