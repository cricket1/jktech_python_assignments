## PYTORCH IMAGE CLASSIFICATION
This repostiory has both training and inference utilitites for training image classfication in pytorch.
It also has utilities to convert the resulting model to opencv 

### common.py
Common configuration need to be set here.For example
```
dataset = '../data/test_data'
model_dir = '../models'
model_name = 'mask'
visualise_dir = '../visualise'
bs = 3  # batch_size
num_epochs = 3
```

### Training 
The training scripts are 
- main_train_mobilenet.py
- main_train_resnet.py

### Prediction
- main_predict_mobilenet.py
- main_predict_resnet.py

### Conversion to gray
main_create_gray_img_dataset converts color images to gray images stored with 3 channels

### Convert the model to ONNX
- pytorch_to_onnx_resnet.py

### Opencv inference 
- opencv_infer.py
