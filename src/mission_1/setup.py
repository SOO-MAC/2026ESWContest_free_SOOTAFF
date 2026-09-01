from setuptools import setup
from glob import glob
import os

package_name = 'mission_1'

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
    description="SOOMAC Mission 1: table setting (name_tag/snack/bottle placement per seat)",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "task_1_detection_node = mission_1.task_1_detection_node:main",
            "task_1_manager_node = mission_1.task_1_manager_node:main",
            "task_1_arm_control_node = mission_1.task_1_arm_control_node:main",
            "task_1_motor_control_node = mission_1.task_1_motor_control_node:main",
        ],
    },
)
