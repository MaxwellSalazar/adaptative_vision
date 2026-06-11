"""
simulation.launch.py
=====================
Launch principal: levanta Gazebo + UR5 + cámara + todos los nodos VS.

Uso:
    ros2 launch adaptive_visual_servo simulation.launch.py scenario:=static
    ros2 launch adaptive_visual_servo simulation.launch.py scenario:=linear mode:=proposed
    ros2 launch adaptive_visual_servo simulation.launch.py scenario:=sinusoidal mode:=baseline

Argumentos:
    scenario    : static | linear | sinusoidal  (default: static)
    mode        : proposed | baseline           (default: proposed)
    lambda_s    : ganancia imagen               (default: 0.5)
    lambda_tau  : peso penalización torque      (default: 0.1)
    gui         : true | false                  (default: true)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    ExecuteProcess, TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('adaptive_visual_servo')

    # ── Argumentos ─────────────────────────────────────────────────────────
    scenario_arg = DeclareLaunchArgument(
        'scenario', default_value='static',
        description='Escenario: static | linear | sinusoidal')
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='proposed',
        description='Controlador: proposed | baseline')
    lambda_s_arg = DeclareLaunchArgument(
        'lambda_s', default_value='0.5',
        description='Ganancia de convergencia imagen')
    lambda_tau_arg = DeclareLaunchArgument(
        'lambda_tau', default_value='0.1',
        description='Peso de penalizacion de torque')
    gui_arg = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Mostrar GUI de Gazebo')

    scenario = LaunchConfiguration('scenario')
    mode = LaunchConfiguration('mode')
    lambda_s = LaunchConfiguration('lambda_s')
    lambda_tau = LaunchConfiguration('lambda_tau')
    gui = LaunchConfiguration('gui')

    # ── Gazebo ─────────────────────────────────────────────────────────────
    world_file = os.path.join(pkg, 'worlds', 'vs_arena.world')
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            PythonExpression(["'", world_file, "'"]),
            PythonExpression(["'' if '", gui, "' == 'true' else '-s'"]),
        ],
        output='screen',
    )

    # ── Robot state publisher ───────────────────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                os.path.join(pkg, '..', '..', '..', 'share',
                             'adaptive_visual_servo', 'urdf', 'ur5_camera.urdf')
            ).read() if os.path.exists(
                os.path.join(pkg, '..', '..', '..', 'share',
                             'adaptive_visual_servo', 'urdf', 'ur5_camera.urdf')
            ) else '',
        }],
        output='screen',
    )

    # ── Nodos VS (con timer para esperar que Gazebo arranque) ───────────────
    feature_tracker = TimerAction(
        period=5.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='feature_tracker',
            name='feature_tracker',
            parameters=[os.path.join(pkg, 'config', 'tracker.yaml')],
            output='screen',
        )]
    )

    depth_estimator = TimerAction(
        period=5.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='depth_estimator',
            name='depth_estimator',
            parameters=[os.path.join(pkg, 'config', 'depth.yaml')],
            output='screen',
        )]
    )

    # Controlador propuesto
    adaptive_controller = TimerAction(
        period=6.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='adaptive_vs',
            name='adaptive_vs_controller',
            parameters=[{
                'lambda_s': lambda_s,
                'lambda_tau': lambda_tau,
                'baseline_mode': False,
            }],
            condition=UnlessCondition(
                PythonExpression(["'", mode, "' == 'baseline'"])),
            output='screen',
        )]
    )

    # Controlador baseline
    baseline_controller = TimerAction(
        period=6.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='ibvs_controller',
            name='ibvs_baseline_controller',
            parameters=[{
                'lambda_s': lambda_s,
            }],
            condition=IfCondition(
                PythonExpression(["'", mode, "' == 'baseline'"])),
            output='screen',
        )]
    )

    # Target publisher
    target_pub = TimerAction(
        period=5.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='target_publisher',
            name='target_publisher',
            parameters=[{'scenario': scenario}],
            output='screen',
        )]
    )

    # Energy monitor
    energy_mon = TimerAction(
        period=6.0,
        actions=[Node(
            package='adaptive_visual_servo',
            executable='energy_monitor',
            name='energy_monitor',
            parameters=[{
                'experiment_name': PythonExpression(
                    ["'", scenario, "_", mode, "'"]),
            }],
            output='screen',
        )]
    )

    return LaunchDescription([
        scenario_arg, mode_arg, lambda_s_arg, lambda_tau_arg, gui_arg,
        gazebo,
        robot_state_pub,
        feature_tracker,
        depth_estimator,
        adaptive_controller,
        baseline_controller,
        target_pub,
        energy_mon,
    ])
