from setuptools import setup, find_packages

setup(
    name='Phoenix',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Phoenix',
    author_email='',
    description='',
    include_package_data=True,
    install_requires=[
        'mahjong==1.2.0.dev5',
        'apache-beam==2.28.0',
    ],
    scripts=['pipeline.py']

)
