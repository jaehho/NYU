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

    for node_name, node_config in config_data.items():
        params = node_config.get('ros__parameters', {})

        # Bird detector node
        if node_name.startswith('bird_detector_'):
            cam_id = str(params.get('camera_id', '0'))
            namespace = f'cam{cam_id}'  # Avoid numeric-only namespaces

            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='bird_detector',
                    name=node_name,
                    namespace=namespace,
                    parameters=[params],
                    output='screen'
                )
            )

        # Mic amplitude and spectrogram nodes
        elif node_name.startswith('mic') and node_name.endswith('_node'):
            mic_name = params.get('mic_name', 'micX')
            mic_suffix = mic_name[-1]  # '0' from 'mic0'
            namespace = f'mic{mic_suffix}'

            # Amplitude node
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

            # Spectrogram node (reuses mic params)
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

            mic_plot_topics.append(f'/{namespace}/audio_amplitude/data')

        # React behavior node (motor trigger)
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

    # Launch background reset GUI
    launch_actions.append(
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    )

    # rqt_plot for mic amplitude curves
    if mic_plot_topics:
        launch_actions.append(
            ExecuteProcess(
                cmd=['ros2', 'run', 'rqt_plot', 'rqt_plot', *mic_plot_topics],
                output='screen',
                shell=False
            )
        )

    # rqt_console for logs
    launch_actions.append(
        ExecuteProcess(
            cmd=['ros2', 'run', 'rqt_console', 'rqt_console'],
            output='screen',
            shell=False
        )
    )

    return LaunchDescription(launch_actions)
