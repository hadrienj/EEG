from setuptools import setup

setup(name='EEG',
      version='0.1',
      description='EEG analysis utils',
      url='https://github.com/hadrienj/EEG',
      author='hadrienj',
      author_email='hadrienjean@gmail.com',
      license='MIT',
      packages=['EEG'],
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'scipy',
          'mne',
          'couchdb',
          'h5py'
      ],
      zip_safe=False)