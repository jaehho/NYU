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
        params = node_config['ros__parameters']

        # Bird detector node
        if node_name.startswith('bird_detector_'):
            cam_id = params.get('camera_id', '0')
            namespace = f'cam{cam_id}'  # Avoid numeric namespace segment

            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='bird_detector',
                    name=node_name,
                    namespace=namespace,
                    parameters=[params],
                    remappings=[
                        ('motion_detected', 'motion_detected'),
                        ('motion_frame', 'motion_frame'),
                        (f'set_background_{cam_id}', 'set_background')
                    ],
                    output='screen'
                )
            )

        # Mic amplitude node
        elif node_name.startswith('mic') and node_name.endswith('_node'):
            mic_name = params.get('mic_name', 'micX')
            mic_suffix = mic_name[-1]  # '0' from 'mic0'
            namespace = f'mic{mic_suffix}'

            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='audio_rms',
                    name=node_name,
                    namespace=namespace,
                    parameters=[params],
                    remappings=[
                        ('audio_amplitude', 'audio_amplitude')
                    ],
                    output='screen'
                )
            )

            launch_actions.append(
                Node(
                    package='budgie_bot',
                    executable='audio_spectrogram',
                    name=f"{mic_name}_spectrogram",
                    namespace=namespace,
                    parameters=[{
                        'mic_name': mic_name,
                        'samplerate': params['samplerate'],
                        'fft_size': 512,
                        'history_len': 100
                    }],
                    output='screen'
                )
            )

            # Track for rqt_plot
            mic_plot_topics.append(f'/{namespace}/audio_amplitude/data')

    # Launch GUI to control camera background resets
    launch_actions.append(
        ExecuteProcess(
            cmd=['python', gui_script],
            output='screen',
            shell=False
        )
    )

    # One rqt_plot for all mic amplitude curves
    if mic_plot_topics:
        launch_actions.append(
            ExecuteProcess(
                cmd=['ros2', 'run', 'rqt_plot', 'rqt_plot', *mic_plot_topics],
                output='screen',
                shell=False
            )
        )

    return LaunchDescription(launch_actions)
