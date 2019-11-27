from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector
import json

checkpoint_file = './train_logs/latest.pth'
score_thr = 0.5

# build the model from a config file and a checkpoint file
test_model = init_detector('./config/retinanet_r50_fpn_1x.py', checkpoint_file)

# test a single image and show the results
ret = []
for img_id in range(1, 13069):
    img = './data/test/' + str(img_id) + '.png'
    result = inference_detector(test_model, img)
    now = {
        'bbox': [],
        'score': [],
        'label': []
    }
    for num in range(len(result)):
        for r in result[num]:
            now['bbox'].append(
                (float(r[1]), float(r[0]), float(r[3]), float(r[2]))
                )
            now['score'].append(float(r[-1]))
            now['label'].append(int(num+1))
    ret.append(now)

with open('0856030.json', 'w') as outfile:
    json.dump(ret, outfile)
