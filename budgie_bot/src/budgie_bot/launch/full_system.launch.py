from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
import yaml
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('budgie_bot')
    config_path = os.path.join(package_dir, 'config', 'system_config.yaml')
    gui_script = os.path.join(package_dir, 'scripts', 'set_bg_gui.py')

    # Load YAML config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    launch_actions = []
    mic_plot_topics = []

    # Iterate through nodes
    for node_name, node_config in config_data.items():
        params = node_config.get('ros__parameters', {})

        # === CAMERA GROUP ===
        if node_name.startswith('cam'):
            cam_id = str(params.get('camera_id', '0'))
            namespace = f'cam{cam_id}'

            # Camera Node
            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='camera',
                    name=node_name,
                    namespace=namespace,
                    parameters=[params],
                    output='screen'
                )
            )

            # Bird Detector Node (suffix match: bird_detector0 for cam0)
            detector_name = f'bird_detector{cam_id}'
            if detector_name in config_data:
                detector_params = config_data[detector_name].get('ros__parameters', {})
                launch_actions.append(
                    Node(
                        package='budgie_bot',
                        executable='bird_detector',
                        name=detector_name,
                        namespace=namespace,
                        parameters=[detector_params],
                        output='screen'
                    )
                )

            # Image Viewer Node for motion_frame
            launch_actions.append(
                Node(
                    package='image_view',
                    executable='image_view',
                    name=f'motion_frame_viewer_{cam_id}',
                    namespace=namespace,
                    remappings=[('image', 'motion_frame')],  # resolves to /camX/motion_frame
                    parameters=[{'image_transport': 'compressed'}],
                    output='screen'
                )
            )

        # === MICROPHONE GROUP ===
        elif node_name.startswith('mic'):
            mic_name = params.get('mic_name', node_name)
            mic_suffix = mic_name[-1]  # assumes mic0, mic1, etc.
            namespace = f'mic{mic_suffix}'

            # Amplitude Node
            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='audio_rms',
                    name=node_name,
                    namespace=namespace,
                    parameters=[params],
                    output='screen'
                )
            )

            # Spectrogram Node
            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='audio_spectrogram',
                    name=f"{mic_name}_spectrogram",
                    namespace=namespace,
                    parameters=[params],
                    output='screen'
                )
            )

            # For rqt_plot
            mic_plot_topics.append(f'/{namespace}/audio_amplitude/data')

        # === REACT BEHAVIOR NODE ===
        elif node_name == 'react_behavior':
            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='react_behavior',
                    name='react_behavior',
                    parameters=[params],
                    output='screen',
                )
            )

    # === GUI for background reset ===
    launch_actions.append(
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    )

    # === rqt_plot for mic amplitude curves ===
    if mic_plot_topics:
        launch_actions.append(
            ExecuteProcess(
                cmd=['ros2', 'run', 'rqt_plot', 'rqt_plot', *mic_plot_topics],
                output='screen',
                shell=False
            )
        )

    # === rqt_console for logs ===
    launch_actions.append(
        ExecuteProcess(
            cmd=['ros2', 'run', 'rqt_console', 'rqt_console'],
            output='screen',
            shell=False
        )
    )

    return LaunchDescription(launch_actions)
