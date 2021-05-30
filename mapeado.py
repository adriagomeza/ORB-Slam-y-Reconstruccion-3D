# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from PIL import Image, ImageFilter, ImageDraw
import open3d as o3d

f=284.21 #focal
T=0.152 #distancia entre camaras
window_size = 45
search_size = 100
perspectiveAngle=60
width=2048 #img width
height=1024 #img height

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


        
"""
Depth_map
"""

def disparity_to_depth_map(disparity_matrix):

	disparity_matrix = f * T / disparity_matrix
	disparity_matrix = disparity_matrix
	return disparity_matrix
"""
Mapeado
"""

def depth_to_cloud(depth, mask_map, mask_2, x, y):
    
    z=depth[mask_map][mask_2] * -1
    points = np.zeros((z.shape[0],3))
    
    anglePos=np.degrees(np.arctan(np.tan(math.radians(0.5*perspectiveAngle))*((2*(x[mask_2]))/(width-1)))) - perspectiveAngle/2
    points[:,1] = z * np.tan(np.radians(anglePos))
    
    angleY=2*math.degrees(math.atan(math.tan(math.radians(perspectiveAngle/2))/(width/height)))
    anglePos=np.degrees(np.arctan(np.tan(math.radians(0.5*angleY))*((2*y[mask_2])/(height-1)))) - angleY/2
    points[:,2] = z * np.tan(np.radians(anglePos))
    
    points[:,0] = z
    
    return points

def match_clouds(x, y, alfa, points):
    
    alfa = math.radians(alfa)
    
    R = np.array([[math.cos(alfa), - math.sin(alfa)],
                  [math.sin(alfa), math.cos(alfa)]])
    
    p = np.transpose(R @ np.transpose(points[:,[0,1]]))
    
    points[:,0] = p[:,0]
    points[:,1] = p[:,1]
    
    return points

def uniformar(disparity, x, y, w):
    
    for i in range(x.shape[0]):
        if y[i] > w and y[i] < height - w and x[i] > w and x[i] < width - w:
            window = disparity[y[i]-w:y[i]+w, x[i]-w:x[i]+w]
            k, l = np.where(window > 0)
            m = np.mean(window[k,l])
            disparity[y[i], x[i]] = m
    return disparity

def mapper3d(imgL, imgR, mapbot, mapa, colores):
    
    imgL = cv2.medianBlur(imgL,19)
    imgR = cv2.medianBlur(imgR,19)
    imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgRGB = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(imgLgray,imgRgray)
    # plt.imshow(disparity,'gray')
    # plt.show()
    mask_map = disparity > 1
    y, x = np.where(mask_map==True)
    disparity = uniformar(disparity, x, y, 5)
    # plt.imshow(disparity,'gray')
    # plt.show()
    depth = disparity_to_depth_map(disparity)
    mask_2 = depth[mask_map] < 0.5
    points = depth_to_cloud(depth, mask_map, mask_2, x, y)
    colors = imgRGB[mask_map][mask_2]
    # create_output(points, colors, 'reconstructed.ply')
    if mapa.shape[0]>1:
        points = match_clouds(mapbot.x, mapbot.y, mapbot.orientacion, points)
    mapa_points = np.concatenate((mapa, points), axis=0)
    mapa_colors = np.concatenate((colores, colors), axis=0)
    
    return mapa_points, mapa_colors

    
def scattered_map(kp):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(kp[:,0], kp[:,1], kp[:,2], color = "green")
    plt.show()
    
    