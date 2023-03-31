from setuptools import setup

setup(name='pt_img_classify',
      version='0.1',
      description='mask detection in a face train and test ',
      author='sumant',
      packages=['pt_img_classify','pt_img_classify.train_utils',
                'pt_img_classify.infer'],
      include_package_data=True
      )