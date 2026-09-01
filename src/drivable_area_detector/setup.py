from setuptools import setup
import os
from glob import glob

package_name = 'drivable_area_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yml')),
        (os.path.join('share', package_name, 'config'), ['drivable_area.rviz']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'scikit-learn',
        'scikit-image',
        'open3d',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='83450047+vdessenn@users.noreply.github.com',
    description='Détection zone circulable LiDAR Pandora',
    license='MIT',
    entry_points={
        'console_scripts': [
            # === v03 Minimal Pipeline (3 nodes: 01→02→RANSAC) ===
            # Path: /patchworkpp/ground → [01] → /road_surface → [02] → /ring_boundaries → [RANSAC] → markers
            '01_road_points_extraction_node = drivable_area_detector.01_road_points_extraction:main',
            '02_ring_boundary_extraction_node = drivable_area_detector.02_ring_boundary_extraction:main',
            'road_markers_publisher = drivable_area_detector.rviz2_publisher:main',
            # REMOVED: 03_curves_smoothing_node, 04_marker_array_node, voxelisation_node
        ],
    },
)