from setuptools import find_packages, setup
from glob import glob
import os

package_name = "gui"

data_files = [
    (
        "share/ament_index/resource_index/packages",
        ["resource/" + package_name],
    ),
    (
        "share/" + package_name,
        ["package.xml"],
    ),
]

if os.path.isdir("images"):
    data_files.append(
        (
            "share/" + package_name + "/images",
            glob("images/*"),
        )
    )

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer='박종훈',
    maintainer_email='jh2kevin@gmail.com',
    description="행사 보조 로봇 GUI",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gui = gui.gui:main",
        ],
    },
)
