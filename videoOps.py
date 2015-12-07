import numpy as np
import cv2
import DIAToolbox as dia


def LOGDetector(image):
	
	l13 = -image[0:image.shape[0]-4,2:image.shape[1]-2]
	l22 = -image[1:image.shape[0]-3,1:image.shape[1]-3]
	l23 = -image[1:image.shape[0]-3,2:image.shape[1]-2]
	l24 = -image[1:image.shape[0]-3,3:image.shape[1]-1]
	l31 = -image[2:image.shape[0]-2,0:image.shape[1]-4]
	l32 = -image[2:image.shape[0]-2,1:image.shape[1]-3]
	l33 = image[2:image.shape[0]-2,2:image.shape[1]-2]
	l34 = -image[2:image.shape[0]-2,3:image.shape[1]-1]
	l35 = -image[2:image.shape[0]-2,4:image.shape[1]]
	l42 = -image[3:image.shape[0]-1,1:image.shape[1]-3]
	l43 = -image[3:image.shape[0]-1,2:image.shape[1]-2]	
	l44 = -image[3:image.shape[0]-1,3:image.shape[1]-1]
	l53 = -image[4:image.shape[0],2:image.shape[1]-2]
	
	return l13+l22+2*l23+l24+l31+2*l32+16*l33+2*l34+l35+l42+2*l43+l44+l53

def Laplacian(image):
	
	l11 = -image[0:image.shape[0]-2,0:image.shape[1]-2]
	l12 = -image[0:image.shape[0]-2,1:image.shape[1]-1]
	l13 = -image[0:image.shape[0]-2,2:image.shape[1]]
	l21 = -image[1:image.shape[0]-1,0:image.shape[1]-2]
	l22 = image[1:image.shape[0]-1,1:image.shape[1]-1]
	l23 = -image[1:image.shape[0]-1,2:image.shape[1]-0]
	l31 = -image[2:image.shape[0],0:image.shape[1]-2]
	l32 = -image[2:image.shape[0],1:image.shape[1]-1]
	l33 = -image[2:image.shape[0],2:image.shape[1]-0]
	
	return l11+l12+l13-2*l21+2*l22-2*l23+l31+l32+l33
	
def gaussianPyramid(image):
	
	pyramid = {}
	resized = [1.0,0.5,0.25,0.125,0.0625]
	
	for i in range(0,4):
		
		temp = []
		
		rgray = cv2.resize(image, (0,0), fx=resized[i], fy=resized[i])
		temp.append(cv2.GaussianBlur(rgray, (3,3), 0))
		temp.append(cv2.GaussianBlur(rgray, (5,5), 0))
		temp.append(cv2.GaussianBlur(rgray, (7,7), 0))
		temp.append(cv2.GaussianBlur(rgray, (9,9), 0))
		temp.append(cv2.GaussianBlur(rgray, (11,11), 0))
		
		pyramid[i] = temp
	
	return pyramid

def differenceOfGaussian(pyramid):
	
	DOGPyramid = {}
	
	for i in range(len(pyramid)):
		
		x = pyramid[i]
		temp = []
		
		temp.append(x[0]-x[1])
		temp.append(x[1]-x[2])
		temp.append(x[2]-x[3])
		temp.append(x[3]-x[4])
		
		DOGPyramid[i] = temp
		
	return DOGPyramid

def getKeypoints(dog):
	
	top = dog[2][0]
	mid = dog[2][1]
	bot = dog[2][2]
	
	keyPoints = []
	for i in range(1,mid.shape[0]-1):
		for j in range(1,mid.shape[1]-1):
			
			maxRef = np.array((top[i-1:i+2,j-1:j+2],mid[i-1:i+2,j-1:j+2],bot[i-1:i+2,j-1:j+2]))
			maxRef[1,1] = 0
	
			minRef = np.array((top[i-1:i+2,j-1:j+2],mid[i-1:i+2,j-1:j+2],bot[i-1:i+2,j-1:j+2]))
			minRef[1,1] = 255
			
			locMax = np.amax(maxRef)
			locMin = np.amin(minRef)
			
			locMin = np.amin(np.array((top[i-1:i+2,j-1:j+2],mid[i-1:i+2,j-1:j+2],bot[i-1:i+2,j-1:j+2])))
					
			if (locMax < mid[i][j] or locMin > mid[i][j]):
					
				keyPoints.append((i,j))
					
	return keyPoints
	
def detectCorners(gray, frame):
	
	corners = cv2.goodFeaturesToTrack(gray,1000,0.01,20)
	corners = np.int0(corners)
	
	for i in corners:
		x,y = i.ravel()
		cv2.circle(frame, (x,y), 3, 255, -1)

cap = cv2.VideoCapture(0)
switch = True
edgeImage = []

while(True):

	ret, frame = cap.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gPyrm = gaussianPyramid(gray)
	DOG = differenceOfGaussian(gPyrm)
	
	kp = getKeypoints(DOG)
	grayResize = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)
	
	for i in kp:

		cv2.circle(grayResize, i, 3, 255, -1)
	#cv2.imshow('og',frame)
	cv2.imshow('gray',grayResize)
	#cv2.imshow('DOG', DOG[0][0])
	#cv2.imshow('DOG2', DOG[0][1])
	#cv2.imshow('DOG3', DOG[0][2])
	#cv2.imshow('DOG4', DOG[0][3])
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
