from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_name = 'rs_bottom'
    return LaunchDescription([
        Node(
            package='realsense_infra_hdr_merge', 
            executable='realsense_infra_hdr_merge_node',
            parameters=[ { "saturation_weight": 1.0,
                           "exposure_weight" : 10.0, # give preference to exposure
                           "contrast_weight" : 1.0
                        }],
            name='realsense_infra_hdr_merge',
            remappings=[('~/camera/image',f'/{camera_name}/camera/infra2/image_rect_raw'),
                        ('~/camera/info',f'/{camera_name}/camera/infra2/camera_info'),
                        ('~/camera/metadata',f'/{camera_name}/camera/infra2/metadata')]
        )
    ])
