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
	args = cmd.parse()

	vin, W, H, fps, _4cc, nframes = vid.istream(args['input'])
	if args['output']:
		fout = args['output']
	else:
		fout = args['input'].split('.')[0] + '-' + args['tracker'] + '.avi'
	vout = vid.ostream(fout, W, H, fps, _4cc)

	red, capas = yolo.load()

	if args['tracker'] == 'sort':
		tracker = mot.Sort()
	elif args['tracker'] == 'cent':
		tracker = ctd.CentroidTracker(args['maxrem'])
	else:
		print('Tiene que pasar un método de rastreo')
		quit()

	if not args['predict']:
		# Create an accumulator that will be updated during each frame
		acc = mm.MOTAccumulator(auto_id=True)
		if args['output']:
			gt = open(args['gt'])
		else:
			gt = open(args['input'].split('.')[0] + '.gt')
		stats = open('stats.dat', 'a')

	mem = defaultdict(list)
	cols = post.paleta()
	unavez = True
	am = open(args['input'].split('.')[0] + '.am', 'w')
	hyp = open(args['input'].split('.')[0] + '.' + args['tracker'], 'w')

	n = nframes
	if args['nframes'] and args['nframes'] < nframes:
		n = args['nframes']
	for i in range(n):
		status, frame = vin.read()

		# fin de video
		#if not grabbed:
		#	break

		#pre.filtro(frame)

		ti_d = time.time()
		dets = yolo.detect(frame, red, capas, args['confidence'], args['threshold'], W, H)
		tf_d = time.time()

		for det in dets:
			am.write(f'000 {" ".join([str(int(d)) for d in det[:-1]])} ')
		am.write('\n')

		if False:
			elap = end - start
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * nframes))
			unavez = False

		ti_t = time.time()
		boxes = tracker.update(dets)
		tf_t = time.time()

		if args['tracker'] == 'sort':
			boxes = post.box2cent(frame, boxes)
		elif args['tracker'] == 'cent':
			pass

		frame = post.trace(frame, boxes, mem, cols)
		vout.write(frame)

		tr_ids = list(boxes.keys())
		tr_cents = list(boxes.values())
		s = []
		for id_, ct in zip(tr_ids, tr_cents):
			cx = ct[0]
			cy = ct[1]
			s.append(f'"{id_}": [{cx},{cy}]')
		hyp.write('{'+', '.join(s)+'}\n')

		if not args['predict']:
			#tr_ids, tr_cents = mota.render(boxes)
			#gt_ids, gt_cents = mota.render2(gt.readline())
			gts = json.loads(gt.readline())
			gt_ids = list(map(int, gts.keys()))
			gt_cents = list(gts.values())
			mota.update(acc, gt_ids, gt_cents, tr_ids, tr_cents, args['maxdist'])

		ttot = time.time()
		print(f'{int(100*(i+1)/n):3}% ({i+1}/{n}) {tf_d-ti_d:.2}s (yolo)'
			  f" + {(tf_t-ti_t)*1000:.2}ms ({args['tracker']}) = {ttot-ti_d:.2}s  ", end='\r')

	print()
	if not args['predict']:
		mota.compute(acc, stats, args['input'].split('.')[0] + args['tracker'])
		gt.close()
		stats.close()

	vout.release()
	vin.release()
	hyp.close()

main()
