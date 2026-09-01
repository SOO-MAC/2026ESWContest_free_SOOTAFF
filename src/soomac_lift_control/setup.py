import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'soomac_lift_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kim',
    maintainer_email='kim@example.com',
    description='SOOMAC lift control package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lift_serial_server = soomac_lift_control.lift_serial_server:main',
            'lift_film_node = soomac_lift_control.lift_film_node:main',
            
        ],
    },
)
