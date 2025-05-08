from setuptools import setup, find_packages


setup(name='PISL',
      version='1.0.0',
      description='Unsupervised Learning of Intrinsic Semantics With Diffusion Model for Person Re-Identification',
      author='Xuefeng Tao',
      author_email='taoxuefeng@stu.jiangnan.edu.cn',
      url='https://github.com/taoxuefong/Diffusion-reid',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages()
      )