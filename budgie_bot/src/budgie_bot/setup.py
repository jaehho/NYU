from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'budgie_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            [f for f in glob('launch/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'scripts'),
            [f for f in glob('scripts/*.py') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'config'),
            [f for f in glob('config/*.yaml') if os.path.isfile(f)]),
    ],
    install_requires=[
        'setuptools',
        'matplotlib',
        'opencv-python',
        'cv_bridge',
        'pyfirmata2',
    ],
    zip_safe=True,
    maintainer='Jaeho Cho',
    maintainer_email='jaeho2025@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera = budgie_bot.camera:main',
            'bird_detector = budgie_bot.bird_detector:main',
            'audio_rms = budgie_bot.audio_rms:main',
            'audio_spectrogram = budgie_bot.audio_spectrogram:main',
            'react_behavior = budgie_bot.react_behavior:main',
        ],
    },
)
