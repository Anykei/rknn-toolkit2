import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
# Model from https://github.com/airockchip/rknn_model_zoo
# ONNX_MODEL = 'yolov5s_relu.onnx'
# RKNN_MODEL = 'yolov5s_relu.rknn'
# IMG_PATH = './bus.jpg'
# DATASET = './dataset.txt'

ONNX_MODEL = './face_detection_yolo/runs/detect/train3/weights/best.onnx'
RKNN_MODEL = '_x.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = list(["person"])

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


count = 0
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[128, 128, 128]], std_values=[[128, 128, 128]], target_platform='rk3588',optimization_level=1,
                 quantized_algorithm='kl_divergence')#,
                # quantized_method='channel', quantized_algorithm='mmse')1
    print('done')

    # Load ONNX model
    print('--> Loading model')
    # ret = rknn.load_onnx(model=ONNX_MODEL)
    ret = rknn.load_tflite(model="best_float32.tflite")
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Accuracy analysis')

    # ret = rknn.accuracy_analysis(inputs=['/home/an_nemenko/repo/tmp/rknn-toolkit2/rknn-toolkit2/examples/onnx/yolov5/bus.jpg'], output_dir=None)
    # if ret != 0:
    #     print('Accuracy analysis failed!')
    #     exit(ret)
    # print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # ret = rknn.init_runtime(target='rk3588', perf_debug=True)
    # perf_detail = rknn.eval_perf()
    # print(perf_detail)

    def show(img):
        cv2.imshow("image", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return


    img_path = './drones'
    img_files = os.listdir(img_path)
    num_img_files = len(img_files)
    print(f'Количество файлов в папке с изображениями: {num_img_files}')

    for img_file in img_files:
        frame = cv2.imread(os.path.join(img_path, img_file))

        # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Inference
        print('--> Running model')
        # img2 = np.expand_dims(img, 0)
        outputs = rknn.inference(inputs=[img], data_format=['nhwc'])

        outputs = outputs[0]

        # show_outputs(softmax(outputs))

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        length = max((IMG_SIZE, IMG_SIZE))
        scale = length / 640

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(np.array(box)*640)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            draw_bounding_box(
                img,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )
        if len(boxes) > 0:
            show(img)
        else:
            count += 1
            print(count)

    cv2.destroyAllWindows()
    rknn.release()
