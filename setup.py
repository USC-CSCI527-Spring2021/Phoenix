from setuptools import setup, find_packages

setup(
    name='Phoenix',
    version='',
    packages=find_packages(),
    url='',
    license='',
    author='Phoenix',
    author_email='',
    description='',
    include_package_data=True,
    install_requires=[
        'tensorflow_transform==0.28.0',
        'tensorflow==2.4.1',
        'mahjong==1.2.0.dev5',
        'tensorboard==2.4.1',
        'apache-beam==2.28.0',
    ],
    scripts=['pipeline.py']

)
