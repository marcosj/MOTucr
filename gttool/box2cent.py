import numpy as np
import sys

if len(sys.argv) != 2:
	print('Uso:', sys.argv[0], 'gtboxes')
	sys.exit()

outf = sys.argv[1][:sys.argv[1].rfind('.')] + '.gt'
with open(sys.argv[1]) as gt2, open(outf, 'w') as gt:
	for cuadro in gt2:
		objs = np.array(cuadro.split()).reshape(-1,5)
		s = []
		for obj in objs:
			id_, xmin, ymin, xmax, ymax = obj
			cx = int((int(xmin) + int(xmax)) / 2)
			cy = int((int(ymin) + int(ymax)) / 2)
			s.append(f'"{id_}": [{cx},{cy}]')
		gt.write('{'+', '.join(s)+'}\n')
