from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='FasterKModes', 
    packages=['FasterKModes'], 

    version='0.1.2', 

    license='MIT', 

    install_requires=['numpy', 'scipy', 'appdirs'], 

    author='NaoMatch', 
    author_email='mn1491625@gmail.com', 

    url='https://github.com/NaoMatch/FasterKModes', 

    description='Faster Implementation of KModes and KPrototypes.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    keywords='KModes KPrototypes Machine-Learning Clustering', 

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],

    package_data={
        # your_package_name 配下の extras フォルダ内の全ファイルを含める
        'FasterKModes': ['src/*'],
    },
)
