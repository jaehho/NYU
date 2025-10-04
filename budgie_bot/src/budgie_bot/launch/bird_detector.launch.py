from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # List of camera IDs to launch
    camera_ids = ['0']

    # Package and GUI script paths
    package_dir = get_package_share_directory('budgie_bot')
    gui_script = os.path.join(package_dir, 'scripts', 'set_bg_gui.py')

    launch_actions = []

    for cid in camera_ids:
        namespace = f'cam{cid}'

        # Camera Node
        launch_actions.append(
            Node(
                package='budgie_bot',
                executable='camera',
                name=f'camera_{cid}',
                namespace=namespace,
                parameters=[{
                    'camera_id': cid,
                    'fps': 30.0
                }],
                output='screen'
            )
        )

        # Bird Detector Node
        launch_actions.append(
            Node(
                package='budgie_bot',
                executable='bird_detector',
                name=f'bird_detector_{cid}',
                namespace=namespace,
                parameters=[{
                    'camera_id': cid,
                    'detector_name': 'bg_subtract',
                    'detection_fps': 5.0,
                    'min_motion_area': 500
                }],
                output='screen'
            )
        )

        # Image Viewer Node for motion_frame (compressed)
        launch_actions.append(
            Node(
                package='image_view',
                executable='image_view',
                name=f'motion_frame_viewer_{cid}',
                namespace=namespace,
                remappings=[('image', f'/{namespace}/motion_frame')],
                parameters=[{
                    'image_transport': 'compressed'
                }],
                output='screen'
            )
        )

    # Background Reset GUI
    launch_actions.append(
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    )

    return LaunchDescription(launch_actions)
