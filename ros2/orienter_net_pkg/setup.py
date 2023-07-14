from setuptools import setup

package_name = 'orienter_net_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sakoda',
    maintainer_email='shintaro.sakoda@tier4.jp',
    description='The package of OrienterNet',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "orienter_net_node = orienter_net_pkg.orienter_net_node:main"
        ],
    },
)
