import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
	print('Uso:', sys.argv[0], 'video', 'bboxes')
	sys.exit()

vin = cv2.VideoCapture(sys.argv[1])

with open(sys.argv[2]) as gt:
	cv2.namedWindow('gt', cv2.WINDOW_NORMAL)
	while True:
		grabbed, img = vin.read()

		# fin de video
		if not grabbed:
			break

		objs = np.array(gt.readline().strip().split())
		objs = objs.reshape(-1,5)
		#print(objs)

		for obj in objs:
			xmin, ymin, xmax, ymax = list( map( int, obj[1:5] ) )
			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
			cv2.putText(img, ' '.join(obj[1:5]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
		cv2.imshow('gt', img)
		k = cv2.waitKey(0)
		if k%256 == 27:
			break
	cv2.destroyWindow('gt')
vin.release()
