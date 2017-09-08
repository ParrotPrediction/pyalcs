from setuptools import setup, find_packages

setup(name='gym_maze',
      version='0.0.2',
      description='Maze environments for OpenAI Gym Environment',
      keywords='acs lcs machine-learning reinforcement-learning openai',
      url='https://github.com/ParrotPrediction/openai-maze-envs',
      author='Parrot Prediction',
      author_email='contact@parrotprediction.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'gym',
          'networkx'
      ],
      include_package_data=False,  # We don't have other types of files
      zip_safe=False)
