import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()
img_labels = ["Enojo", "Felicidad", "Felicidad", "Felicidad", "Felicidad", "Felicidad","Neutral","Neutral","Tristeza","Enojo","Sorpresa","Disgusto","Felicidad","Enojo"]
correct = 0

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size = 48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized


def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo_1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        # TODO: Para la imagen de entrada, utiliza tu modelo para predecir la clase mas probale
        pred_label = EMOTIONS_MAP[modelo.predict(transformed.unsqueeze(0)).numpy().argmax().item()]

        # Original / transformada
        # pred_label (str): nombre de la clase predicha
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

        if(pred_label == img_labels[img_title_paths[path]]):
            global correct
            correct += 1

    # Imprimir el accuracy 
    accuracy = correct / len(img_title_paths) * 100
    print(f"Accuracy: {accuracy}%")

if __name__=="__main__":
    # Direcciones relativas a este archivo
    img_paths = ["./test_imgs/hacegaba.jpg", "./test_imgs/happy.png", "./test_imgs/happyA.png", "./test_imgs/happyz.jpg", "./test_imgs/hz.jpg", "./test_imgs/jony.png", "./test_imgs/NeutralA.png", "./test_imgs/neutralZ.png", "./test_imgs/sadZ.jpg", "./test_imgs/seriusZ.jpg", "./test_imgs/surprisedZ.jpg", "./test_imgs/zambrano.png", "./test_imgs/zambri.jpg", "./test_imgs/angryZ.jpg"]
    predict(img_paths)