import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'my_scout_control'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config',
            ),
            glob('config/*.yaml'),
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='parkjonghoon',
    maintainer_email='parkjonghoon@todo.todo',

    description=(
        'Scout Mini autonomous navigation '
        'and motion safety control'
    ),

    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            'scout_control = '
            'my_scout_control.scout_control_node:main',

            'velocity_smoother = '
            'my_scout_control.velocity_smoother:main',

            'cmd_vel_gate = '
            'my_scout_control.cmd_vel_gate:main',
        ],
    },
)
