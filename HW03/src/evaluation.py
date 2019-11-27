from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector

checkpoint_file = './train_logs/epoch_25.pth'
score_thr = 0.5

# build the model from a config file and a checkpoint file
test_model = init_detector('./config/retinanet_r50_fpn_1x.py', checkpoint_file)

img = './data/test/29.png'
result = inference_detector(test_model, img)
for num in range(len(result)):
    if result[num].shape[0] != 0:
        for r in result[num]:
            print('label: %2d, score: %.3f' % (num+1, r[-1]))
show_result_pyplot(img, result, test_model.CLASSES, score_thr=score_thr)
