import pyrealsense2 as rs
import numpy as np

def StartRealSense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("RGB Camera Not Found")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.infra, 960, 540, rs.format.y8, 30)
    else:
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

    align_to = rs.stream.color #see https://github.com/IntelRealSense/librealsense/issues/2481
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)
    return(pipeline, align)


