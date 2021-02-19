from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import pyautogui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.3, 0.5, 0.5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--imagen_inicial", type=str, help="Imagen inicial")
    parser.add_argument("--screen_video", type=str, help="Grabacion de pantalla")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    lista = []
    # frame2 = cv2.imread('data/custom/images/frame21.jpg')
    frame2 = cv2.imread(opt.imagen_inicial)
    cont = 1
    ban = 1
    if(opt.screen_video == 1):
        cap = True
    while cap:
        ret, frame = cap.read()
        # if(opt.screen_video == 1):
        img = pyautogui.screenshot() # tomamos un pantallazo
        frame = np.array(img) # convertimos la imagen a un arreglo de numeros
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertimos la imagen BGR a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:800, :800]
        # frame = rgb2gray(frame)
        # else:
            
        if(cont %10 == 0 and ban):
            frame2 = frame
            ban=False
        cont = cont + 1
        if ret is False:
            break
        frame = cv2.resize(frame, (980, 660), interpolation=cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, (980, 660), interpolation=cv2.INTER_CUBIC)
        
        #LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg=Convertir_RGB(frame)
        RGBimg2=Convertir_RGB(frame2)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)


        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    if(int(cls_pred)==0):
                        frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                        cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                        cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    if(int(cls_pred)==1):
                        # output = frame2.copy()
                        # frame2 = cv2.rectangle(frame2, (x1, y1 + box_h), (x2, y1), color, -1)
                        a=7
                        b=20
                        print(float(x1),'  ',x2,'  ',y1 + box_h,'  ',y1,'  ')
                        for i in range(int(y1 ),int(y2)):
                            for j in range(int(x1),int(x2)):
                                dis =abs(frame.item(i, j, 1) - frame2.item(i, j, 1))
                                if( dis>=a and dis<=b):
                                    frame2.itemset((i,j,0),255)
                                    frame2.itemset((i,j,2),255)


                    # frame2 = cv2.addWeighted(frame2, 0.5, output, 1 - .5, 0, output)
                    # cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                    # cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
        #
        #Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        
        if opt.webcam==1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            out.write(RGBimg)
        else:
            
            out.write(Convertir_BGR(RGBimg))
            cv2.imshow('frame', RGBimg)
            out.write(Convertir_BGR(RGBimg2))
            cv2.imshow('frame2', RGBimg2)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
