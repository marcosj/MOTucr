import numpy as np
import cv2

# MEMORIA	centro anterior
#memory = {}
#line = []
#counter = 0

def paleta():
	np.random.seed(42)
	return np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def box2cent(frame, boxes):
	objs = {}
	for box in boxes:
		#(x, y) = (int(box[0]), int(box[1]))
		#(w, h) = (int(box[2]), int(box[3]))
		#cv2.rectangle(frame, (x, y), (w, h), 3, 2)
		#cv2.imshow('aa', frame)
		#cv2.waitKey()
		xmin, ymin, xmax, ymax, id_ = box
		x, y = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
		objs[int(id_)] = (x, y)
	return objs

def trace(frame, objs, mem, paleta):
	#objs = dict(objs)
	for id_, ct in objs.items():
		mem[id_].append(ct)

	for id_ in mem:
		tray = [tuple(t) for t in mem[id_]]
		color = [int(c) for c in paleta[id_ % len(paleta)]]
		col2 = [255-c for c in color]
		cv2.circle(frame, tray[0], 3, col2, 1, cv2.LINE_AA)
		for i in range(1, len(tray)):
			cv2.line(frame, tray[i-1], tray[i], color, 1, cv2.LINE_AA)
			cv2.circle(frame, tray[i], 3, col2, 1, cv2.LINE_AA)
		cv2.rectangle(frame, (tray[-1][0]-12, tray[-1][1]-6),
			(tray[-1][0]+2, tray[-1][1]-24), col2, -1)
		cv2.putText(frame, str(id_), (tray[-1][0]-10, tray[-1][1]-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

	return frame
'''
	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = defaultdict(list)
	#line = [(43, 543), (550, 655)]

	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			color = [int(c) for c in paleta[indexIDs[i] % len(paleta)]]
			#cv2.rectangle(frame, (x, y), (w, h), color, 2)

			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				#cv2.line(frame, p0, p1, color, 3)

				#if intersect(p0, p1, line[0], line[1]):
				#	counter += 1

			# text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}".format(indexIDs[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			i += 1

	# draw line
	# cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

	# draw counter
	# cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
	# counter += 1

	# saves image file
	#cv2.imwrite("output/frame-{:02}.png".format(frameIndex), frame)
	return frame, memory

# Return true if line segments AB and CD intersect
#def intersect(A,B,C,D):
#	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#def ccw(A,B,C):
#	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
'''
