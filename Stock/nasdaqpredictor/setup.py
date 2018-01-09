from setuptools import setup, find_packages

setup(name='Nasdaq Predictor',
      version='1.0',
      description='Predicts stock stuff',
      author='Faragó Balázs',
      author_email='minimalgeek@gmail.com',
      packages=find_packages(exclude=['tests'])
      )
