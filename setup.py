# %%
from setuptools import setup, find_packages

# %%
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# %%
setup(name='bioio',
      version='0.2.0',
      description='BioIO',
      url='http://github.com/mhorlacher/bioio',
      author='Marc Horlacher',
      author_email='marc.horlacher@gmail.com',
      license='MIT',
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=False,
      entry_points = {
            'console_scripts': [
                  'bioio=bioio.__main__:main',
            ],
      },
      zip_safe=False)
