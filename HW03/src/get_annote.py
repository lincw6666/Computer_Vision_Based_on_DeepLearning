import numpy as np
import pandas as pd
import cv2
import h5py
import mmcv


# Get arguments from user.
parser = ArgumentParser()
parser.add_argument(
    '-d', '--dir',
    help='The directory where you store all the training images.',
    dest='train_dir'
)
args = parser.parse_args()
if args.train_dir is None:
    print('Error!! Expect 1 arguments!!\n')
    parser.print_help()
    sys.exit(0)


# Get the image name from the .mat file.
def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


# Get the bounding boxes from the .mat file.
def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = np.array(values)
    return attrs


# Get the image size from the .mat file.
def get_size(img_id):
    return cv2.imread(args.train_dir + str(img_id) + '.png').shape[:2]


annote = []
fp = h5py.File('./data/train/digitStruct.mat', 'r')
for img_id in range(fp['/digitStruct/bbox'].shape[0]):
    tmp = {}
    tmp['filename'] = get_name(img_id, fp)
    tmp['height'], tmp['width'] = get_size(img_id+1)
    bbox = get_bbox(img_id, fp)
    tmp['labels'] = bbox['label'].astype(np.int64)
    tmp['bboxes'] = np.stack(
        [bbox['left'], bbox['top'], bbox['width'], bbox['height']],
        -1).astype(np.float32)
    annote.append(tmp)
mmcv.dump(annote, './SVHN.pkl')
