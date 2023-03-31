import torch
import os
from pt_img_classify.train_utils.common import model_dir, model_name, num_epochs
from pt_img_classify.train_utils.common_mobilenet import predict

model = torch.load(os.path.join(model_dir, model_name + '_model_' + str(num_epochs-1) + '.pt'))
print('')
predict(model, '../tests/test_imgs/test_imgs_gray/mask_sumant.jpg')
print('')
predict(model, '../tests/test_imgs/test_imgs_gray/nomask_sumant.jpg')
