# -*- coding: utf-8 -*-
import numpy as np
import math
import sim 
import time

D=0.38
Perim=0.325689

def connect(port):

    sim.simxFinish(-1)
    clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5)
    if clientID == 0: print("conectado a", port)
    else: print("no se pudo conectar")
    return clientID
    
class robot:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.orientacion = 0
        self.keypoints = []


    def act_move(self, d, angulo):
        
        if d == 0:
            self.orientacion += angulo
            
            if self.orientacion>360:
                self.orientacion = self.orientacion % 360
            elif self.orientacion < 0:
                self.orientacion = self.orientacion + 360
        
        else:
            if self.orientacion >= 0 and self.orientacion < 90:
                self.x += d * math.cos(self.orientacion)
                self.y += d * math.sin(self.orientacion)
                
            elif self.orientacion >= 90 and self.orientacion < 180:
                self.x -= d * math.cos(self.orientacion-90)
                self.y += d * math.sin(self.orientacion-90)
                
            elif self.orientacion >= 180 and self.orientacion < 270:
                self.x -= d * math.cos(self.orientacion-180)
                self.y -= d * math.sin(self.orientacion-180)
                
            else:
                self.x += d * math.cos(self.orientacion-270)
                self.y -= d * math.sin(self.orientacion-270)
    
    def move(self, d, angulo, clientID, ruedaL, ruedaR):
        
        if d==0:
            rot = abs((angulo*np.pi*D)/360)
            angFin = (rot/Perim) * 360
            if angulo < 0:
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, math.radians(-angFin),sim.simx_opmode_streaming)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, math.radians(angFin),sim.simx_opmode_streaming)
                time.sleep(1.3)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, 0,sim.simx_opmode_streaming)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, 0,sim.simx_opmode_streaming)
                
            else:
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, math.radians(angFin),sim.simx_opmode_streaming)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, math.radians(-angFin),sim.simx_opmode_streaming)
                time.sleep(1.3)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, 0,sim.simx_opmode_streaming)
                retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, 0,sim.simx_opmode_streaming)
                
        else:
            m = (d / Perim) * 360
            retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, math.radians(m),sim.simx_opmode_streaming)
            retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, math.radians(m),sim.simx_opmode_streaming)
            time.sleep(1)
            retCode = sim.simxSetJointTargetVelocity(clientID, ruedaL, 0,sim.simx_opmode_streaming)
            retCode = sim.simxSetJointTargetVelocity(clientID, ruedaR, 0,sim.simx_opmode_streaming)
            
        self.act_move(d, angulo)

    
    
    