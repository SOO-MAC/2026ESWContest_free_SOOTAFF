from setuptools import setup
from glob import glob
import os

package_name = 'mission_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryeong',
    maintainer_email='ryeong@todo.todo',
    description='SOOMAC Mission 2: hand/mic detection + Scout approach + arm delivery task',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "task_2_detection_node = mission_2.task_2_detection_node:main",
            "task_2_manager_node = mission_2.task_2_manager_node:main",
            "task_2_manager_node2 = mission_2.task_2_manager_node2:main",
            "task_2_arm_control_node = mission_2.task_2_arm_control_node:main",
            "task_2_motor_control_node = mission_2.task_2_motor_control_node:main",
        ],
    },
)
