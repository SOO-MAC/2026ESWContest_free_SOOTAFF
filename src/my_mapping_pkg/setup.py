import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_mapping_pkg'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='parkjonghoon',
    maintainer_email='parkjonghoon@todo.todo',

    description='Event assistant robot mapping and table waypoint generation',

    # 팀 저장소의 실제 라이선스에 맞게 최종 제출 전에 수정
    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            'table_mapper = my_mapping_pkg.table_mapper:main',
        ],
    },
)
