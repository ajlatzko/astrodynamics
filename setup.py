from setuptools import setup

VERSION = '0.6.0'
DESCRIPTION = 'Collection of functions used in EAS 4510 and beyond'

setup(name='astrodynamics',
      version=VERSION,
      description=DESCRIPTION,
      url='',
      author='Alexander Latzko',
      author_email='<ajlatzko@gmail.com>',
      packages=['astrodynamics'],
      install_requires=['numpy'],
      zip_safe=False)