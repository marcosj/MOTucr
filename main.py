import time
import json
from collections import defaultdict
import parser as cmd
import video as vid
import yolo
import rast1
import cent as ctd
import sort.sort as mot
import preproc as pre
import postproc as post
import mota
import motmetrics as mm

def main():
	# Parses command line arguments
	args = cmd.parse()

	# Opens input video and output with trajectory
	vin, W, H, fps, _4cc, nframes = vid.istream(args['input'])
	if args['output']:
		fout = args['output']
	else:
		fout = args['input'].split('.')[0] + '-' + args['tracker'] + '.avi'
	vout = vid.ostream(fout, W, H, fps, _4cc)

	# Initializes yolo
	red, capas = yolo.load()

	# Initializes tracker
	if args['tracker'] == 'sort':
		tracker = mot.Sort()
	elif args['tracker'] == 'cent':
		tracker = ctd.CentroidTracker(args['maxrem'])
	else:
		print('Tiene que pasar un método de rastreo')
		quit()

	# Initializes metric and opens ground truth data
	if not args['predict']:
		# Create an accumulator that will be updated during each frame
		acc = mm.MOTAccumulator(auto_id=True)
		if args['output']:
			gt = open(args['gt'])
		else:
			gt = open(args['input'].split('.')[0] + '.gt')
		stats = open('stats.dat', 'a')

	# Centroid memory
	mem = defaultdict(list)
	# Random colors
	cols = post.paleta()

	# Annotation files
	am = open(args['input'].split('.')[0] + '.am', 'w')
	hyp = open(args['input'].split('.')[0] + '.' + args['tracker'], 'w')

	# number of frames to process
	n = nframes
	if args['nframes'] and args['nframes'] < nframes:
		n = args['nframes']

	for i in range(n):
		# Reads frame from stream
		status, frame = vin.read()

		# Applies filter to image to improve detection
		#pre.filtro(frame)

		# Detects objects with Yolo. Takes over 99% of run time
		ti_d = time.time()
		dets = yolo.detect(frame, red, capas, args['confidence'], args['threshold'], W, H)
		tf_d = time.time()

		# Writes bounding boxes to file
		for det in dets:
			am.write(f'000 {" ".join([str(int(d)) for d in det[:-1]])} ')
		am.write('\n')

		# Determines identities with tracker
		ti_t = time.time()
		boxes = tracker.update(dets)
		tf_t = time.time()

		# Converts output to dictionary for Sort
		if args['tracker'] == 'sort':
			boxes = post.box2cent(frame, boxes)

		# Draws object trayectories
		frame = post.trace(frame, boxes, mem, cols)
		vout.write(frame)

		# Writes tracker output to file
		tr_ids = list(boxes.keys())
		tr_cents = list(boxes.values())
		s = []
		for id_, ct in zip(tr_ids, tr_cents):
			cx = ct[0]
			cy = ct[1]
			s.append(f'"{id_}": [{cx},{cy}]')
		hyp.write('{'+', '.join(s)+'}\n')

		# Updates metrics
		if not args['predict']:
			gts = json.loads(gt.readline())
			gt_ids = list(map(int, gts.keys()))
			gt_cents = list(gts.values())
			mota.update(acc, gt_ids, gt_cents, tr_ids, tr_cents, args['maxdist'])

		ttot = time.time()
		print(f'{int(100*(i+1)/n):3}% ({i+1}/{n}) {tf_d-ti_d:.2}s (yolo)'
			  f" + {(tf_t-ti_t)*1000:.2}ms ({args['tracker']}) = {ttot-ti_d:.2}s  ", end='\r')
	print()

	# Computes total metrics
	if not args['predict']:
		mota.compute(acc, stats, args['input'].split('.')[0] + args['tracker'])
		gt.close()
		stats.close()

	# Housekeeping
	vout.release()
	vin.release()
	hyp.close()

main()
