Running the tracker in predict mode to generate ground truth data:

	python3 main.py -i example/pav.mp4 -r sort -n 60

outputs:

- Video with trajectory `pav-sort.avi`
- Detected bounding boxes `pav.am`
- Tracker output `pav.sort`

Now you can annotate the video with the help of

	python3 gttool/gttool.py example/pav.mp4 example/pav.am

and add ids to a copy of `pav.am`, namely `pav.gt2`.

Next you transform it to json

	python3 gttool/box2cent.py example/pav.gt2

is `pav.gt`.

Now you can run the tracker in metrics mode

	python3 main.py -i example/pav.mp4 -r sort -p 0 -n 60

to get, aditionally:

- MOT metrics `stats.dat`

