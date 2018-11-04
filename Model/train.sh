time ../../build/tools/caffe train \
	-solver  solver.prototxt \
	-gpu all
	-weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
	#-snapshot ./SSDH48_iter_2000.solverstate
	# -weights ./SSDH48_iter_original.caffemodel