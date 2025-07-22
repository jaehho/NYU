from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    camera_ids = ['0']

    package_dir = get_package_share_directory('budgie_bot')
    gui_script = os.path.join(package_dir, 'scripts', 'set_bg_gui.py')

    launch_actions = []

    for cid in camera_ids:
        launch_actions.append(
            Node(
                package='budgie_bot',
                executable='bird_detector',
                name=f'camera_node_{cid}',
                parameters=[
                    {'camera_id': cid,
                     'inference_fps': 5.0,
                     'min_motion_area': 500,
                     'detection_mode': 'bg_subtract'}
                ],
                output='screen'
            )
        )

    launch_actions.append(
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    )

    return LaunchDescription(launch_actions)
