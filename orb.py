import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt
from time import time
from skimage.feature import plot_matches
from skimage.transform import pyramid_gaussian
from PIL import Image, ImageFilter, ImageDraw


def FAST(img, N=9, threshold=0.15, nms_window=2, train=False, ID3=None):
    kernel = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]])/16
    
    img = convolve2d(img, kernel, mode='same')

    cross_idx = np.array([[3,0,-3,0], [0,3,0,-3]])
    circle_idx = np.array([[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3],
	                       [0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]])
    circle_aux = np.array([[3,2,1,-1,-2,-3,-3,-2,-1,1,2,3],
	                       [1,2,3,3,2,1,-1,-2,-3,-3,-2,-1]])

    corner_img = np.zeros(img.shape)
    keypoints = []
    if ID3==None:
        n=N-4
        for y in range(12, img.shape[0]-12,2):
            for x in range(12, img.shape[1]-12,2):
                Ip = img[y,x]
                t = threshold*Ip if threshold < 1 else threshold
                if np.sum(Ip+t < img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3:
                    if np.sum(img[y+circle_aux[0,:], x+circle_aux[1,:]] >= Ip+t) >= n:
                        corner_img[y,x] = np.sum(np.abs(Ip - img[y+circle_idx[0,:], x+circle_idx[1,:]]))
                        keypoints.append(([x,y],corner_img[y,x]))
                elif np.sum(Ip-t > img[y+cross_idx[0,:], x+cross_idx[1,:]]) >= 3:
                    if np.sum(img[y+circle_aux[0,:], x+circle_aux[1,:]] <= Ip-t) >= n:
                        corner_img[y,x] = np.sum(np.abs(Ip - img[y+circle_idx[0,:], x+circle_idx[1,:]]))
                        keypoints.append(([x,y],corner_img[y,x]))
    else:
        pass
    
    keypoints=best(keypoints)
    keypoints, score = zip(*keypoints)
    keypoints = list(keypoints)
    
    if nms_window != 0:
        fewer_kps = []
        for [x, y] in keypoints:
            window = corner_img[y-nms_window:y+nms_window+1, x-nms_window:x+nms_window+1]
            # v_max = window.max()
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)

def ordenar(e):
    return e[1]

def best(corners):
    corners.sort(reverse=True, key=ordenar)
    return corners[:400]

def corner_orientations(img, corners):
    OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
    OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
            OFAST_MASK[15 + j, 15 + i] = 1
    mrows, mcols = OFAST_MASK.shape
    mrows2 = int((mrows - 1) / 2)
    mcols2 = int((mcols - 1) / 2)
    
    img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    orientations = []
    for i in range(corners.shape[0]):
        c0, r0 = corners[i, :]
        m01, m10 = 0, 0
        for r in range(mrows):
            m01_temp = 0
            for c in range(mcols):
                if OFAST_MASK[r,c]:
                    I = img[r0+r, c0+c]
                    m10 = m10 + I*(c-mcols2)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp*(r-mrows2)
        orientations.append(np.arctan2(m01, m10))

    return np.array(orientations)


def BRIEF(img, keypoints, orientations=None, n=256, patch_size=9, sigma=1, sample_seed=42):
    
    random = np.random.RandomState(seed=sample_seed)
    img = cv2.GaussianBlur(img,(5,5),2)

    samples = random.randint(-(patch_size - 2) // 2 +1, (patch_size // 2), (n * 2, 2))
    samples = np.array(samples, dtype=np.int32)
    pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    if orientations is None:
        mask = (  ((patch_size//2 - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - patch_size//2 + 1))
                & ((patch_size//2 - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - patch_size//2 + 1)))

        keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for p in range(pos1.shape[0]):
            pr0 = pos1[p, 0]
            pc0 = pos1[p, 1]
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]
            for k in range(keypoints.shape[0]):
                kr = keypoints[k, 1]
                kc = keypoints[k, 0]
                if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                    descriptors[k, p] = True
    else:
        distance = int((patch_size//2)*1.5)
        mask = (  ((distance - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - distance + 1))
                & ((distance - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - distance + 1)))

        keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)
        orientations = np.array(orientations[mask], copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for i in range(descriptors.shape[0]):
            angle = orientations[i]
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)
            
            kr = keypoints[i, 1]
            kc = keypoints[i, 0]
            for p in range(pos1.shape[0]):
                pr0 = pos1[p, 0]
                pc0 = pos1[p, 1]
                pr1 = pos2[p, 0]
                pc1 = pos2[p, 1]
                
                spr0 = round(sin_theta*pr0 + cos_theta*pc0)
                spc0 = round(cos_theta*pr0 - sin_theta*pc0)
                spr1 = round(sin_theta*pr1 + cos_theta*pc1)
                spc1 = round(cos_theta*pr1 - sin_theta*pc1)

                if img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]:
                    descriptors[i, p] = True
    return descriptors


def match(kp1, kp2, descriptors1, descriptors2, distance_ratio=None):
    
    distances = cdist(descriptors1, descriptors2, metric='hamming')
    matches = []
    num = kp1.shape[0]
    for k in range(num):
        aux = np.where(kp2[:,1] == kp1[k,1])[0]
        if aux.shape[0] != 0:
            arg = np.argmin(distances[k, aux])
            arg = aux[arg]
            matches.append(np.append(kp1[k,:], kp2[arg,0]))
            
    return np.array(matches)

def match2(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = cdist(descriptors1, descriptors2, metric='hamming')   
    
    indices1 = np.arange(descriptors1.shape[0])     
    indices2 = np.argmin(distances, axis=1)         
                                                    

    matches1 = np.argmin(distances, axis=0)     
    mask = indices1 == matches1[indices2]       
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches

def test():
    N_LAYERS = 3
    DOWNSCALE = 1.5
    
    original_img1 = img1 = Image.open('tests/mochila/im0.png')
    im1 = np.array(img1)
    gray1 = img1.convert('L')
    gray1 = np.array(gray1)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))
    
    original_img2 = img2 = Image.open('tests/mochila/im1.png')
    im2 = np.array(img2)
    gray2 = img2.convert('L')
    gray2 = np.array(gray2)
    grays2 = list(pyramid_gaussian(gray2, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))
    
    scales = [DOWNSCALE**i for i in range(N_LAYERS+1)]
    print('scales: ', scales, '\n')
    features_img1 = Image.fromarray(np.copy(im1))
    features_img2 = Image.fromarray(np.copy(im2))
    
    kps1 = []
    kps2 = []
    
    ds1 = []
    ds2 = []
    
    for i in range(1, N_LAYERS+1):
        print('pyramid layer: ', i)
        print('scales[i]: ', scales[i])
        scale_coeff1 = (gray1.shape[1]/grays1[i].shape[1], gray1.shape[0]/grays1[i].shape[0])
        scale_coeff2 = (gray2.shape[1]/grays2[i].shape[1], gray2.shape[0]/grays2[i].shape[0])
        
        print('scale_coeff1: ', scale_coeff1)
        print('scale_coeff2: ', scale_coeff2)
        
        print('grays1[i] shape: ', grays1[i].shape)
        print('grays2[i] shape: ', grays2[i].shape)
        
        scale_kp1 = FAST(grays1[i], N=12, threshold=0.2, nms_window=3)
        scale_kp2 = FAST(grays2[i], N=12, threshold=0.2, nms_window=3)
        
        print('kp1: ', len(scale_kp1))
        print('kp2: ', len(scale_kp2))
        
        for keypoint in scale_kp1:
   
            x0 = np.round(keypoint*scale_coeff1)[0]-7*scales[i]
            y0 = np.round(keypoint*scale_coeff1)[1]-7*scales[i]
            x1 = np.round(keypoint*scale_coeff1)[0]+7*scales[i]
            y1 = np.round(keypoint*scale_coeff1)[1]+7*scales[i]
            draw1 = ImageDraw.Draw(features_img1)
            draw1.ellipse([x0,y0,x1,y1], outline='red', width=1)
        for keypoint in scale_kp2:
   
            x0 = np.round(keypoint*scale_coeff2)[0]-7*scales[i]
            y0 = np.round(keypoint*scale_coeff2)[1]-7*scales[i]
            x1 = np.round(keypoint*scale_coeff2)[0]+7*scales[i]
            y1 = np.round(keypoint*scale_coeff2)[1]+7*scales[i]
            draw2 = ImageDraw.Draw(features_img2)
            draw2.ellipse([x0,y0,x1,y1], outline='red', width=1)
        # plt.figure(figsize=(20,10))
        # plt.subplot(1,2,1)
        # plt.imshow(grays1[i], cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(features_img1)
    
        # plt.figure(figsize=(20,10))
        # plt.subplot(1,2,1)
        # plt.imshow(grays2[i], cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(features_img2)
        
        # orientations1 = corner_orientations(grays1[i], scale_kp1)
        # orientations2 = corner_orientations(grays2[i], scale_kp2)
        
        kps1.append(np.round(scale_kp1*scale_coeff1).astype(np.int32))
        kps2.append(np.round(scale_kp2*scale_coeff2).astype(np.int32))
            
        d1 = BRIEF(grays1[i], scale_kp1, orientations=None, n=256)
        d2 = BRIEF(grays2[i], scale_kp2, orientations=None, n=256)
        ds1.append(d1)
        ds2.append(d2)
        
        plt.show()
        print('-'*50)
    
    
    plt.figure(figsize=(20,10))
    plt.imshow(features_img1)
    
    plt.figure(figsize=(20,10))
    plt.imshow(features_img2)
    plt.show()
    
    scale_kps1 = np.vstack(kps1)
    scale_kps2 = np.vstack(kps2)
    
    scale_ds1 = np.vstack(ds1)
    scale_ds2 = np.vstack(ds2)
    
    scale_ms = match2(scale_kps1, scale_kps2, scale_ds1, scale_ds2, distance_ratio=None)
    
    print('total matches: ', len(scale_ms))
    
    fig = plt.figure(figsize=(20.0, 30.0))
    ax = fig.add_subplot(1,1,1)
    plot_matches(ax, im1, im2, np.flip(scale_kps1, 1), np.flip(scale_kps2, 1), scale_ms[:100], alignment='horizontal', only_matches=True)



def orb(imgL, imgR, N_LAYERS=3, DOWNSCALE=1.5, N=12, t=0.2, n=256):    
    
    imL = cv2.resize(imgL, (512,256), interpolation = cv2.INTER_AREA)
    imR = cv2.resize(imgR, (512,256), interpolation = cv2.INTER_AREA)
    
    features_img1 = Image.fromarray(np.copy(imL))
    features_img2 = Image.fromarray(np.copy(imR))
    
    grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    graysL = list(pyramid_gaussian(grayL, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))
    
    grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
    graysR = list(pyramid_gaussian(grayR, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))
    
    kpsL = []
    kpsR = []
    
    dsL = []
    dsR = []
    
    scales = [DOWNSCALE**i for i in range(N_LAYERS+1)]
    
    for i in range(N_LAYERS):
        scale_coeffL = (grayL.shape[1]/graysL[i].shape[1], grayL.shape[0]/graysL[i].shape[0])
        scale_coeffR = (grayR.shape[1]/graysR[i].shape[1], grayR.shape[0]/graysR[i].shape[0])
        
        kpL = FAST(graysL[i], N=N, threshold=t, nms_window=2)
        kpR = FAST(graysR[i], N=N, threshold=t, nms_window=2)
        kpsL.append(np.round(kpL*scale_coeffL).astype(np.int32))
        kpsR.append(np.round(kpR*scale_coeffR).astype(np.int32))
        
        # for keypoint in kpL:
   
        #     x0 = np.round(keypoint*scale_coeffL)[0]-3*scales[i]
        #     y0 = np.round(keypoint*scale_coeffL)[1]-3*scales[i]
        #     x1 = np.round(keypoint*scale_coeffL)[0]+3*scales[i]
        #     y1 = np.round(keypoint*scale_coeffL)[1]+3*scales[i]
        #     draw1 = ImageDraw.Draw(features_img1)
        #     draw1.ellipse([x0,y0,x1,y1], outline='red', width=1)
            
        # for keypoint in kpR:
   
        #     x0 = np.round(keypoint*scale_coeffR)[0]-3*scales[i]
        #     y0 = np.round(keypoint*scale_coeffR)[1]-3*scales[i]
        #     x1 = np.round(keypoint*scale_coeffR)[0]+3*scales[i]
        #     y1 = np.round(keypoint*scale_coeffR)[1]+3*scales[i]
        #     draw2 = ImageDraw.Draw(features_img2)
            # draw2.ellipse([x0,y0,x1,y1], outline='red', width=1)
        
        # orientationsL = corner_orientations(graysL[i], kpL)
        # orientationsR = corner_orientations(graysR[i], kpR)
            
        dL = BRIEF(graysL[i], kpL, orientations=None, n=n)
        dR = BRIEF(graysR[i], kpR, orientations=None, n=n)
        dsL.append(dL)
        dsR.append(dR)
    
    # plt.figure(figsize=(20,10))
    # plt.imshow(features_img1)
    
    # plt.figure(figsize=(20,10))
    # plt.imshow(features_img2)
    # plt.show()
        
    kpsL = np.vstack(kpsL)
    kpsR = np.vstack(kpsR)
    
    dsL = np.vstack(dsL)
    dsR = np.vstack(dsR)
    
    # scale_ms = match2(kpsL, kpsR, dsL, dsR, distance_ratio=None)
    
    # print('total matches: ', len(scale_ms))
    
    # fig = plt.figure(figsize=(20.0, 30.0))
    # ax = fig.add_subplot(1,1,1)
    # plot_matches(ax, imL, imR, np.flip(kpsL, 1), np.flip(kpsR, 1), scale_ms, alignment='horizontal', only_matches=True)
    
    matches = match(kpsL, kpsR, dsL, dsR, distance_ratio=None)
    
    return matches
    

if __name__ == "__main__":
    
    test()