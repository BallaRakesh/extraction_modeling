'''

for training yolov7-----

python train.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --cfg yolov7.yaml --weights yolov7.pt --name yolov7_custom --hyp data/hyp.scratch.custom.yaml --epochs 100

/home/khushal/Downloads/yolov7_env/bin/python train.py --workers 8 --device cpu --batch-size 4 --data data/coco.yaml --cfg yolov7.yaml --weights yolov7.pt --name yolov7_custom --hyp data/hyp.scratch.custom.yaml --epochs 5

usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--rect] [--resume [RESUME]]
                [--nosave] [--notest] [--noautoanchor] [--evolve] [--bucket BUCKET] [--cache-images] [--image-weights] [--device DEVICE] [--multi-scale] [--single-cls] [--adam] [--sync-bn]
                [--local_rank LOCAL_RANK] [--workers WORKERS] [--project PROJECT] [--entity ENTITY] [--name NAME] [--exist-ok] [--quad] [--linear-lr] [--label-smoothing LABEL_SMOOTHING]
                [--upload_dataset] [--bbox_interval BBOX_INTERVAL] [--save_period SAVE_PERIOD] [--artifact_alias ARTIFACT_ALIAS] [--freeze FREEZE [FREEZE ...]] [--v5-metric]

--------------------------------------------------------------------------------------------------------------------------------------------

for inference-----

/home/khushal/Downloads/yolov7_env/bin/python detect.py --weights runs/train/yolov7_weights/weights/best.pt --source /home/khushal/Desktop/Projects/YOLO_V7_CODE/data/test/images  --save-txt --save-conf

usage: detect.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE] [--img-size IMG_SIZE] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES] [--device DEVICE] [--view-img] [--save-txt]
                 [--save-conf] [--nosave] [--classes CLASSES [CLASSES ...]] [--agnostic-nms] [--augment] [--update] [--project PROJECT] [--name NAME] [--exist-ok] [--no-trace]

--------------------------------------------------------------------------------------------------------------------------------------------

for testing-----
python test.py --img-size 640 --batch-size 1 --data data/coco.yaml --weights runs/train/yolov7_custom51/weights/best.pt --save-txt --task test --augment --verbose --save-hybrid --save-conf --save-json --v5-metric

usage: test.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--data DATA] [--batch-size BATCH_SIZE] [--img-size IMG_SIZE] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES] [--task TASK]
               [--device DEVICE] [--single-cls] [--augment] [--verbose] [--save-txt] [--save-hybrid] [--save-conf] [--save-json] [--project PROJECT] [--name NAME] [--exist-ok] [--no-trace]
               [--v5-metric]

'''
