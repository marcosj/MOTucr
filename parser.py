import argparse

def parse():
	# Parse the arguments
	ap = argparse.ArgumentParser(
		description='Draws tracking trajectory of vehicles and outputs MOTA metric.')
	ap.add_argument('-i', '--input', type=str, required=True,
		help='path to input video')
	ap.add_argument('-o', '--output', type=str,
		help='path to output video')
	ap.add_argument('-y', '--yolo', type=str, default='yolo',
		help='trained YOLO directory (default: yolo)')
	ap.add_argument('-c', '--confidence', type=float, default=0.5,
		help='minimum probability to filter weak detections')
	ap.add_argument('-t', '--threshold', type=float, default=0.3,
		help='threshold for supressing duplicate detections')
	ap.add_argument('-r', '--tracker', type=str, required=True,
		help='tracking algorithm: sort o cent')
	ap.add_argument('-x', '--maxrem', type=int, default=5,
		help='number of frames with oclusion for cent')
	ap.add_argument('-g', '--gt', type=str,
		help='manual annotation of input vid')
	ap.add_argument('-p', '--predict', type=int, default=1,
		help='whether to determine metric with ground truth or just trace trajectory. {0,1} Default: 0 (no metric)')
	ap.add_argument('-d', '--maxdist', type=int, default=17,
		help='max distance between consecutive ccentroids for same ID')
	ap.add_argument('-n', '--nframes', type=int,
		help='number of frames to process')
	return vars(ap.parse_args())
