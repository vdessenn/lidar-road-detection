from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'drivable_area_navigator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Victor Dessenne',
    maintainer_email='83450047+vdessenn@users.noreply.github.com',
    description='Real-time navigation using drivable area boundaries from LiDAR detection',
    license='MIT',
    entry_points={
        'console_scripts': [
            'boundary_receiver_node = drivable_area_navigator.boundary_receiver_node:main',
        ],
    },
)
