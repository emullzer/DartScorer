from ultralytics import YOLO

#v8n pour nano, modèle le plus rapide
model = YOLO('yolov8n-pose.pt') 

#entraine le modèle et créé un fichier avec le modele entrainé --> a utiliser pour la detection de fléchettes
model.train(
    data='darts_dataset/data.yaml', 
    epochs=100,      
    imgsz=640,       
    batch=16,  
    device=0, #signifie que je possede une carte graphique NVIDIA     
    name='darts_model' 
)

#pour le modele de la cible

model.train(
    data='dartboards_dataset/data.yaml', 
    epochs=50,      
    imgsz=640,       
    batch=16,  
    device=0, #signifie que je possede une carte graphique NVIDIA     
    name='dartboards_model' 
)