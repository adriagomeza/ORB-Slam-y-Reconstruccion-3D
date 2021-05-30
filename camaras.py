# -*- coding: utf-8 -*-
import sim 
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
 
def escaneo(clientID, sensorHandleL, sensorHandleR):
    
    retCode, resolutionL, imageL=sim.simxGetVisionSensorImage(clientID,sensorHandleL,0,sim.simx_opmode_oneshot_wait)
    retCode, resolutionR, imageR=sim.simxGetVisionSensorImage(clientID,sensorHandleR,0,sim.simx_opmode_oneshot_wait)
    
    imgL=np.array(imageL,dtype=np.uint8)
    imgL.resize([resolutionL[1],resolutionL[0],3])
    imgL = np.flip(imgL, 1)
    # plt.imshow(imgL)
    # plt.show()
    
    imgR=np.array(imageR,dtype=np.uint8)
    imgR.resize([resolutionR[1],resolutionR[0],3])
    imgR = np.flip(imgR, 1)
    # plt.imshow(imgR)
    # plt.show()
    
    return [imgL, imgR]