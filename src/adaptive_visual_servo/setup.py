from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'adaptive_visual_servo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maxwell',
    maintainer_email='maxwell@researcher.edu',
    description='Control visual servo adaptativo con optimización de torque para seguimiento monocular',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ibvs_controller   = adaptive_visual_servo.ibvs_controller:main',
            'adaptive_vs       = adaptive_visual_servo.adaptive_vs_controller:main',
            'feature_tracker   = adaptive_visual_servo.feature_tracker:main',
            'depth_estimator   = adaptive_visual_servo.depth_estimator:main',
            'energy_monitor    = adaptive_visual_servo.energy_monitor:main',
            'target_publisher  = adaptive_visual_servo.target_publisher:main',
        ],
    },
)
