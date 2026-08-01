from setuptools import setup, find_packages

package_name = 'mTIO'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Akin Mert Gumus',
    description='Monocular Thermal-Inertial Odometry (MSCKF)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'run_vo = mTIO.scripts.run_vo:main',
        ],
    },
)