from setuptools import setup, find_packages

package_name = 'thermal_vo'

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
    description='Thermal Monocular Visual Odometry',
    license='MIT',
    entry_points={
        'console_scripts': [
            'run_vo = thermal_vo.scripts.run_vo:main',
        ],
    },
)