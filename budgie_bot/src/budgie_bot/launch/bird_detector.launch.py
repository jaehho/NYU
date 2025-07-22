from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_dir = get_package_share_directory('budgie_bot')
    gui_script = os.path.join(package_dir, 'scripts', 'set_bg_gui.py')

    return LaunchDescription([
        Node(
            package='budgie_bot',
            executable='bird_detector',
            name='bird_detector',
            output='screen',
            parameters=[{
                'camera_ids': ['0'],
                'detection_mode': 'bg_subtract',
                'inference_fps': 5.0,
                'min_motion_area': 500
            }]
        ),
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    ])
