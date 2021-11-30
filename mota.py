import numpy as np
import motmetrics as mm

#acc = mm.MOTAccumulator(auto_id=True) # Create an accumulator that will be updated during each frame

def update(acc, o_ids, o_points, h_ids, h_points, maxdist):
	distances = mm.distances.norm2squared_matrix(o_points, h_points, max_d2=maxdist*maxdist)
	#print("\n-------Distances frame-------\n",distances)

	# Each frame a list of ground truth object / hypotheses ids and pairwise distances
	# is passed to the accumulator. For now assume that the distance matrix given to us.
	# Call update once for per frame. For now, assume distances between frame objects / hypotheses are given.
	frameid = acc.update(
		o_ids,               # Ground truth objects in this frame
		h_ids,                # Detector hypotheses in this frame
		distances             # Distances from objects to hypotheses
	)
	#print("\nFrame ID:",frameid)
	#print("\nacc.events:\n",acc.events) # a pandas DataFrame containing all events
	#print("\nacc.mot_events:\n",acc.mot_events) # a pandas DataFrame containing MOT only events
	#print("\nacc.mot_events.loc[frameid]:\n",acc.mot_events.loc[frameid])


def compute(acc, file, vid):
	# Compute and display metrics
	mh = mm.metrics.create()
	metrics = [
		'num_frames',
		'num_unique_objects',
		'num_misses',
		'num_false_positives',
		'num_switches',
		'num_fragmentations',
		'precision',
		'recall',
		'mota',
		'motp',
		'idf1',
		'idp',
		'idr'
	]
	names = ['#f', 'gt', 'm', 'fp', 'mm', 'frg', 'P', 'R', 'MOTA', '1-MOTP', 'ID-F1', 'ID-P', 'ID-R']
	nmap = {}
	for m, n in zip(metrics, names):
		nmap[m] = n
	summary = mh.compute(acc, metrics=metrics, name=vid)
	strsummary = mm.io.render_summary(
		summary,
		formatters=mh.formatters,
		namemap=nmap
	)
	file.write(strsummary + '\n\n')

'''	# Compute metrics for multiple accumulators or accumulator views
	summary = mh.compute_many(
		[acc, acc.events.loc[0:1]],
		metrics=['num_frames', 'mota', 'motp'],
		names=['full', 'part'])
	print("\nsummary:\n",summary)

	# Reformat column names and how column values are displayed
	strsummary = mm.io.render_summary(
		summary,
		formatters={'mota': '{:.2%}'.format},
		namemap={'mota': 'MOTA', 'motp': 'MOTP'}
	)
	print("\nstrsummary:\n",strsummary)

	# Predefined metric selectors, formatters and metric names
	summary = mh.compute_many(
		[acc, acc.events.loc[0:1]],
		metrics=mm.metrics.motchallenge_metrics,
		names=['full', 'part'])
	strsummary = mm.io.render_summary(
		summary,
		formatters=mh.formatters,
		namemap=mm.io.motchallenge_metric_names
	)
	print("\nstrsummary:\n",strsummary)

	# Overall summary that computes the metrics jointly over all accumulators
	summary = mh.compute_many(
		[acc, acc.events.loc[0:1]],
		metrics=mm.metrics.motchallenge_metrics,
		names=['full', 'part'],
		generate_overall=True
		)
	strsummary = mm.io.render_summary(
		summary,
		formatters=mh.formatters,
		namemap=mm.io.motchallenge_metric_names
	)
	print("\nstrsummary:\n",strsummary)'''
