from setuptools import setup, find_packages

setup(name='pyalcs',
      version='1.0',
      description='Implementation of Anticipatory Learning Classifiers',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/ParrotPrediction/pyalcs',
      author='Parrot Prediction',
      author_email='contact@parrotprediction.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[

      ],
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
