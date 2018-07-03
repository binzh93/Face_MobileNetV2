import sys
sys.path.append('/workspace/mnt/group/face-det/zhubin/caffe/python')
import caffe
caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/workspace/mnt/group/face-det/zhubin/Face_MobileNetV2/solver.prototxt')
solver.net.copy_from('/workspace/mnt/group/face-det/zhubin/train_file/mobilenet_v2.caffemodel')
solver.solve()
