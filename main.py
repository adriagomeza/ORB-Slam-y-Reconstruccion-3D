# -*- coding: utf-8 -*-
from camaras import *
from orb import *
from slam import *
from robot import *
import cv2
import sim 
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

simTime=2

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

if __name__ == "__main__":
    
    clientID = connect(19999)
                             
    if clientID != -1:
        print('Server is connected!')
    else:
        print('Server is unreachable!')
        sys.exit(0)
        
    retCode,camaraL=sim.simxGetObjectHandle(clientID,'Vision_sensorL',sim.simx_opmode_blocking)
    retCode,camaraR=sim.simxGetObjectHandle(clientID,'Vision_sensorR',sim.simx_opmode_blocking)
    retCode,ruedaL=sim.simxGetObjectHandle(clientID,'Motor_L',sim.simx_opmode_blocking)
    retCode,ruedaR=sim.simxGetObjectHandle(clientID,'Motor_R',sim.simx_opmode_blocking)
    retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, 0,sim.simx_opmode_streaming)
    retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, 0,sim.simx_opmode_streaming)
    time.sleep(0.5)
    
    mapbot = robot()
    x=0
    
    mapa = np.zeros((1,3))
    colors = np.zeros((1,3))
    
    while(x <= simTime):
        mapbot, mapa, colors = slam(mapbot, clientID, camaraL, camaraR, ruedaL, ruedaR, mapa, colors)
        x += 1
        
    create_output(mapa, colors, 'reconstructed.ply')