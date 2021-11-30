import cv2 as cv

def initQuitaFondo(frame):
	backSub = cv.createBackgroundSubtractorKNN()
	return backSub

def quitaFondo(frame, backSub):
	return backSub.apply(frame)

def filtro(frame):
	#fgMask = cv.blur(frame, -1, kernel)
	fgMask = cv.GaussianBlur(frame,(5,5),0)
	#fgMask = cv.medianBlur(frame,5)
	#fgMask = cv.bilateralFilter(frame,9,75,75)
	return fgMask

def initRota(angle, W, H):
	return cv.getRotationMatrix2D((W,H), angle, 1)

def rota(frame, R, W, H):
	fgMask = cv.warpAffine(frame, R, (W,H))
	return fgMask
