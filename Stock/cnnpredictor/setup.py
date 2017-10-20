from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return f.readlines()

setup(name='cnnpredictor',
      version='0.1',
      description='Stock trading with CNNs',
      long_description=readme(),
      keywords='deep learning neural network stock trading CNN',
      url='https://github.com/minimalgeek/DeepLearning/tree/master/Stock/cnnpredictor',
      author='Farago Balazs',
      author_email='farago.balazs87@gmail.com',
      license='MIT',
      packages=['cnnpredictor'],
      install_requires=requirements(),
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      include_package_data=True,
      zip_safe=False)
