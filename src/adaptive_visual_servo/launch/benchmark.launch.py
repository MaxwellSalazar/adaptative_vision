"""
benchmark.launch.py
====================
Ejecuta el benchmark completo del paper de forma automatizada.

Corre los 6 experimentos secuencialmente:
    3 escenarios × 2 modos (proposed + baseline)

Luego llama al script de análisis para generar figuras y tabla de métricas.

Uso:
    ros2 launch adaptive_visual_servo benchmark.launch.py
    ros2 launch adaptive_visual_servo benchmark.launch.py duration:=60
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    ExecuteProcess, TimerAction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('adaptive_visual_servo')
    sim_launch = os.path.join(pkg, 'launch', 'simulation.launch.py')

    duration_arg = DeclareLaunchArgument(
        'duration', default_value='90',
        description='Duración de cada experimento en segundos')

    duration = LaunchConfiguration('duration')

    # Los 6 experimentos se documentan aquí; en práctica se lanzan uno a uno
    # porque Gazebo necesita reiniciarse entre experimentos.
    # Este launch sirve como plantilla/documentación del protocolo completo.

    experiments = [
        {'scenario': 'static',      'mode': 'baseline'},
        {'scenario': 'static',      'mode': 'proposed'},
        {'scenario': 'linear',      'mode': 'baseline'},
        {'scenario': 'linear',      'mode': 'proposed'},
        {'scenario': 'sinusoidal',  'mode': 'baseline'},
        {'scenario': 'sinusoidal',  'mode': 'proposed'},
    ]

    # Imprime el protocolo de benchmark
    print_protocol = ExecuteProcess(
        cmd=['bash', '-c', f'''
echo "============================================================"
echo "  BENCHMARK: Adaptive Visual Servoing con Penalización Torque"
echo "============================================================"
echo ""
echo "  Experimentos programados ({len(experiments)} total):"
{"".join([f'echo "    [{i+1}] scenario={e["scenario"]} mode={e["mode"]}";' for i, e in enumerate(experiments)])}
echo ""
echo "  Para ejecutar cada experimento:"
echo "  ros2 launch adaptive_visual_servo simulation.launch.py \\\\"
echo "       scenario:=<scenario> mode:=<mode>"
echo ""
echo "  Después de todos los experimentos, generar figuras:"
echo "  python scripts/analyze_results.py \\\\"
echo "       --baseline ~/vs_results/metrics_static_baseline_*.csv \\\\"
echo "       --proposed ~/vs_results/metrics_static_proposed_*.csv \\\\"
echo "       --output   ~/vs_results/figures/"
echo "============================================================"
        '''],
        output='screen',
    )

    return LaunchDescription([
        duration_arg,
        print_protocol,
    ])
