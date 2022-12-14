from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: MIT License',
    "Operating System :: OS Independent"
]

setup(
     name='tigerhx',
     version='0.0.1',
     description='Processing MRI images based on deep-learning',
     long_description_content_type='text/markdown',
     url='https://github.com/htylab/tigerhx',
     author='Biomedical Imaging Lab, Taiwan Tech',
     author_email='',
     License='MIT',
     classifiers=classifiers,
     keywords='MRI segmentation',
     packages=find_packages(),
     entry_points={
        'console_scripts': [
            'tigerhx = tigerhx.hx:main',
        ]
    },
     python_requires='>=3.7',
     install_requires=[
             'numpy>=1.16.0',
             'nibabel>=3.1.0',
             'scipy>=1.6.0'
         ]
)
