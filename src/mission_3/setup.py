from setuptools import setup
from glob import glob
import os

package_name = 'mission_3'

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
    description='SOOMAC Mission 3: table clean-up (name_tag return, trash/valuables collection)',
    license='Apache License 2.0',
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "task_3_detection_node = mission_3.task_3_detection_node:main",
            "task_3_manager_node = mission_3.task_3_manager_node:main",
            "task_3_arm_control_node = mission_3.task_3_arm_control_node:main",
            "task_3_motor_control_node = mission_3.task_3_motor_control_node:main",
        ],
    },
)
