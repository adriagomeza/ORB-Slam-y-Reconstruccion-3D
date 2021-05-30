# # -*- coding: utf-8 -*-
from orb import *
from camaras import *
from mapeado import *
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt

f=284.21 #focal
T=0.152 #distancia entre camaras
size=0.002 #pixel size
w=512 #img width
h=256 #img height
th=1000000 #distancia a la que girar
mov=1 #movimiento recto
ang=30 #giro del robot
perspectiveAngle=60
Perim=0.325689

"""
    SLAM
"""
def slam(mapbot, clientID, camaraL, camaraR, ruedaL, ruedaR, mapa, colors):
    
    imgs = escaneo(clientID, camaraL, camaraR)
    keypoints = orb(imgs[0], imgs[1], N_LAYERS=3, DOWNSCALE=1.5, N=8, t=0.2, n=256)
    z, keypoints=calcular_profundidad(keypoints)

    kp_3d = keypoints_3d(keypoints, z)
    kp_3d = transformar_keypoints(mapbot.x, mapbot.y, mapbot.orientacion, kp_3d)
    
    if len(mapbot.keypoints) == 0:
        mapbot.keypoints = kp_3d
    else:
        mapbot.keypoints = np.concatenate((mapbot.keypoints, kp_3d), axis=0)
    
    mapa, colors = mapper3d(imgs[0], imgs[1], mapbot, mapa, colors)
    
    d, angulo = decidir_trayectoria(mapbot)
    mapbot.move(d, angulo, clientID, ruedaL, ruedaR)
    return mapbot, mapa, colors


def decidir_trayectoria(mapbot):
    
    alfa = mapbot.orientacion
    boolean = np.zeros(np.shape(mapbot.keypoints[:,[0,1]]))
    aux = np.zeros(np.shape(mapbot.keypoints[:,[0,1]]))
    aux[:,0] = mapbot.keypoints[:,0] - mapbot.x
    aux[:,1] = mapbot.keypoints[:,1] - mapbot.y
    
    if alfa >= 0 and alfa < 90:
        boolean[:,0] = aux[:,0] >= 0 
        boolean[:,1] = aux[:,1] >= 0
        front = np.all(boolean, axis=1)
        
    elif alfa >= 90 and alfa < 180:
        boolean[:,0] = aux[:,0] <= 0 
        boolean[:,1] = aux[:,1] >= 0
        front = np.all(boolean, axis=1)
                
    elif alfa >= 180 and alfa < 270:
        boolean[:,0] = aux[:,0] <= 0 
        boolean[:,1] = aux[:,1] <= 0
        front = np.all(boolean, axis=1)
                
    else:
        boolean[:,0] = aux[:,0] >= 0 
        boolean[:,1] = aux[:,1] <= 0
        front = np.all(boolean, axis=1)
    
    dist = np.sqrt(aux[front,0]**2 + aux[front,1]**2)
    colision = dist < th
    aux = aux[front]
    aux = aux[colision,:]
    
    alfa = np.radians(-alfa)
    R = np.array([[math.cos(alfa), - math.sin(alfa)],
                  [math.sin(alfa), math.cos(alfa)]])
    
    points = np.transpose(R @ np.transpose(aux))
    
    aux[:,0] = points[:,0] < 0.25 
    aux[:,1] = points[:,0] > -0.25
    front = np.all(aux, axis=1)
    points = points[front]
    
    # if len(points) < 10:
    #     return mov, 0
    # else:
    # if len(points)/2 < np.sum(points[:,0] > 0):
    return 0, ang
    # else:
    #     return 0, -ang
    

def transformar_keypoints(x, y, alfa, keypoints):
    
    alfa = math.radians(alfa)
    
    R = np.array([[math.cos(alfa), - math.sin(alfa)],
                  [math.sin(alfa), math.cos(alfa)]])
    
    kp = np.transpose(R @ np.transpose(keypoints[:,[0,1]]))
    
    keypoints[:,0] = kp[:,0] + x
    keypoints[:,1] = kp[:,1] + y
    
    return keypoints
    
def calcular_profundidad(keypoints):
    
    d = abs(keypoints[:,0] - keypoints[:,2])
    out = np.where(d==0)[0]
    d = np.delete(d, out)
    keypoints = np.delete(keypoints, out, axis=0)
    
    ft = f * T
    z = ft / d
    
    return z, keypoints

    
def keypoints_3d(keypoints, z):
    d = abs(keypoints[:,0] - keypoints[:,2])/2
    
    kp = np.zeros(np.shape(keypoints))
    #calculo de X
    
    anglePos=np.degrees(np.arctan(np.tan(math.radians(0.5*perspectiveAngle))*((2*(keypoints[:,0]+d))/(w-1)))) - perspectiveAngle/2
    kp[:,0] = z * np.tan(np.radians(anglePos))
    
    #calculo de Y
    angleY=2*math.degrees(math.atan(math.tan(math.radians(perspectiveAngle/2))/(w/h)))
    anglePos=np.degrees(np.arctan(np.tan(math.radians(0.5*angleY))*((2*keypoints[:,1])/(h-1)))) - angleY/2
    kp[:,2] = z * np.tan(np.radians(anglePos))
    
    kp[:,1] = z
    
    return kp
    
    