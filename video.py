import cv2

def istream(video):
	vs = cv2.VideoCapture(video)
	W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(vs.get(cv2.CAP_PROP_FPS))
	_4cc = int(vs.get(cv2.CAP_PROP_FOURCC))
	n = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	#print(W, H, fps, _4cc, n)
	return vs, W, H, fps, _4cc, n

def ostream(vidfile, W, H, fps, fourcc):
	return cv2.VideoWriter(vidfile, fourcc, fps, (W, H), True)
