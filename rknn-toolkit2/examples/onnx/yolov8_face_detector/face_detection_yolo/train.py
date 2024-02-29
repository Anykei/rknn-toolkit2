from ultralytics import YOLO
# from ultralytics.nn.modules import *

# Re
model = YOLO("yolov8n_custom.yaml").load("yolov8n.pt")  # build a new model from scratch
# model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=10)

metrics = model.val()  # evaluate model performance on the validation set

model.export(format="onnx", opset=12)

# torch.quantization.prepare_qat(model, inplace=True)
# torch.quantization.convert(model, inplace=True)

# model = torch.load("./runs/detect/train5/weights/best.pt")
# model = model['model']
# qua_model = quantize_dynamic(model, dtype=torch.qint8, mapping=None, inplace=False)
#
# torch.save(qua_model, "test.pt")
#
# model = YOLO("test.pt")

# torch.onnx.export(qua_model,  # model being run
#                   (3, 640, 640),  # model input (or a tuple for multiple inputs)
#                   "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
#                   opset_version=12,  # the ONNX version to export the model to
#                   )

# path = model.export(format="onnx", opset=12)  # export the model to ONNX format

# train {}
