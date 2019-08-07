from setuptools import setup, find_packages

setup(name='pyalcs',
      version='1.5',
      description='Implementation of Anticipatory Learning Classifiers',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/ParrotPrediction/pyalcs',
      author='Parrot Prediction Ltd',
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
      packages=find_packages(),
      install_requires=[

      ],
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
