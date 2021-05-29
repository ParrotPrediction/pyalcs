from setuptools import setup, find_packages

testing_requires = [
    'twine>=2.0.0',
    'pytest>=5.2',
    'pytest-cov==2.11.1',
    'pytest-mock==3.5.1',
    'pytest-xdist==2.2.1'
]

mlflow_requires = [
    'boto3',
    'dill',
    'mlflow'
]

docs_requires = [
    'sphinx',
    'nbsphinx',
    'numpydoc'
]

setup(name='pyalcs',
      version='1.6.1',
      description='Implementation of Anticipatory Learning Classifiers',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/ParrotPrediction/pyalcs',
      author='Norbert Kozlowski',
      author_email='nkozlowski@protonmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      project_urls={
          'Source': 'https://github.com/ParrotPrediction/pyalcs',
          'Tracking': 'https://github.com/ParrotPrediction/pyalcs/issues',
      },
      python_requires='>=3.5',
      packages=find_packages(include=['lcs']),
      setup_requires=[
          'pytest-runner',
          'flake8'
      ],
      install_requires=[
          'numpy>=1.17.4',
          'dataslots>=1.0.1'
      ],
      extras_require={
          'mlflow': mlflow_requires,
          'testing': testing_requires,
          'documentation': docs_requires
      },
      test_suite="tests",
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
