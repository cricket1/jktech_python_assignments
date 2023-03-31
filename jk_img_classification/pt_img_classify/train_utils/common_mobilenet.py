from torchvision import transforms
import torch
from PIL import Image

from pt_img_classify.train_utils.common import idx_to_class


def predict(model, test_image_name):
    """
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    """

    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print('predict image {}'.format(test_image_name))
    test_image = Image.open(test_image_name)

    test_image_tensor = transform(test_image)

    # if torch.cuda.is_available():
    #     test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    # else:
    #     test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(2, dim=1)
        for i in range(2):
            print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
                  topk.cpu().numpy()[0][i])
