#converttf2ov.py

import openvino as ov

tflite_model_path = './demo/mobilebert/mobilebert.tflite'
ov_model_path = './model/mobilebert/mobilebert.xml' 

ov_model = ov.convert_model(tflite_model_path)
ov.save_model(ov_model, ov_model_path)
print(f"Model {tflite_model_path} successfully converted and saved to {ov_model_path}")