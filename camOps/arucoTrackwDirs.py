'''
TO USE THIS CODE, GET MAIN() FUNCTION INTO YOUR SCRIPT!
'''
import pyrealsense2 as rs
import numpy as np
import cv2
# import time
# import math
from arucoHelpers_mult import CreateDetector, GetRelativeYaw
from realsenseStartup_mult import StartRealSense
from vectorHelpers_mult import Center

from typing import List, Tuple, Optional, Any

def get_frames(pipeline: rs.pipeline) -> Tuple[rs.frame, rs.depth_frame, rs.video_frame]:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    ir_frame = frames.get_infrared_frame(1)  # 1 for the first infrared sensor
    return frames, depth_frame, ir_frame

def align_frames(frames: rs.frame, align: rs.align) -> Optional[rs.depth_frame]:
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    return aligned_depth_frame

def get_images(depth_frame: rs.depth_frame, ir_frame: rs.video_frame) -> Tuple[np.ndarray, np.ndarray]:
    depth_image = np.asanyarray(depth_frame.get_data())
    ir_image = np.asanyarray(ir_frame.get_data())
    return depth_image, ir_image

def apply_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return depth_colormap

def resize_ir_image(ir_image: np.ndarray, depth_colormap_dim: Tuple[int, int]) -> np.ndarray:
    resized_ir_image = cv2.resize(ir_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    resized_ir_image = cv2.cvtColor(resized_ir_image, cv2.COLOR_GRAY2BGR)
    return resized_ir_image

def detect_markers(arucoimage: np.ndarray, arucoDict: Any, arucoParams: Any) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    corners, ids, rejected = cv2.aruco.detectMarkers(arucoimage, arucoDict, parameters=arucoParams)
    return corners, ids, rejected

def process_markers(corners: List[np.ndarray], ids: List[int], aligned_depth_frame: rs.depth_frame, depth_intrin: rs.intrinsics, arucoimage: np.ndarray) -> np.ndarray:
    markerInfoList = []
    for i in range(len(ids)):
        markerCorners = corners[i][0]
        if markerCorners.size > 0:
            marker_center = Center(markerCorners)
            markedImage = cv2.aruco.drawDetectedMarkers(arucoimage, [corners[i]], np.array([ids[i]]))                    
            cv2.circle(markedImage, tuple(marker_center), 5, (0, 0, 255), cv2.FILLED)
            depth = aligned_depth_frame.get_distance(*marker_center)
            global death_point_in_meters_camera_coords
            depthPoint = rs.rs2_deproject_pixel_to_point(depth_intrin, marker_center, depth)
            angle = GetRelativeYaw(markerCorners)

            # # Print marker information to console for testing/status checking
            # print("\nMarker ID:", ids[i])
            # print("Corners:", markerCorners)
            # print("Center:", marker_center)
            # print("Coordinate in camera frame:", depth_point_in_meters_camera_coords)
            # print("Angle:", angle)

            markerInfo = [ids[i], marker_center, depthPoint, angle]
            markerInfoList.append(markerInfo)

    return markedImage, markerInfoList, depthPoint

def display_image(disp_image: np.ndarray) -> None:
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', disp_image)
    cv2.waitKey(10)

def arucoTrack() -> Tuple: # figure out what kind of outputs we want...float locations of centrepoints?
    (arucoDict, arucoParams, detector) = CreateDetector()
    (pipeline,align) = StartRealSense()

    print('RealSense camera activated, waiting for pipeline...')
    global detection
    imageResized = False
    stepCounter = 0

    try:
        while True:
            frames, depth_frame, ir_frame = get_frames(pipeline)
            if not depth_frame or not ir_frame:
                continue

            depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

            if stepCounter == 0:
                print('Frame pipeline successfully opened...')

            aligned_depth_frame = align_frames(frames, align)
            if aligned_depth_frame is None:
                continue

            depth_image, ir_image = get_images(depth_frame, ir_frame)
            depth_colormap = apply_colormap(depth_image)

            depth_colormap_dim = depth_colormap.shape
            ir_colormap_dim = ir_image.shape

            if depth_colormap_dim != ir_colormap_dim:
                resized_ir_image = resize_ir_image(ir_image, depth_colormap_dim)
                images = np.hstack((resized_ir_image, depth_colormap))
                imageResized = True
            else:
                ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                images = np.hstack((ir_image, depth_colormap))
                imageResized = False

            if imageResized:
                arucoimage = resized_ir_image
            else:
                arucoimage = ir_image

            if len(arucoimage.shape) == 3:
                arucoimage = cv2.cvtColor(arucoimage, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detect_markers(arucoimage, arucoDict, arucoParams)

            if ids is not None and len(ids) > 0:
                markedImage, markerInfoList, depthPoint = process_markers(corners, ids, aligned_depth_frame, depth_intrin, arucoimage)
                detection = 1
                x = depthPoint[0]
                y = depthPoint[1]
                if np.abs(x*100) < 3 and np.abs(y*100) < 3:
                    print("Descend")
                else:
                    if x < 0:
                        print("Move Left: " + str(np.around(np.abs(x*100), 2)) + " cm")
                    if x > 0:
                        print("Move Right: " + str(np.around(np.abs(x*100),2)) + " cm")
                    if y < 0:
                        print("Move Forward: " + str(np.around(np.abs(y*100),2)) + " cm")
                    if y > 0:
                        print("Move Backward: " + str(np.around(np.abs(y*100), 2)) + " cm")
                print('-----------------')

                # Print marker information to console for testing/status checking
                # Structure: [ids[i], marker_center, depth_point_in_meters_camera_coords, angle]
                # print('ID: {}, Center: {}, Depth: {}, Angle: {}'.format(markerInfoList[0][0], markerInfoList[0][1], markerInfoList[0][2], markerInfoList[0][3]))

                markedImage = cv2.cvtColor(markedImage, cv2.COLOR_GRAY2BGR)
                disp_image = np.hstack((markedImage, depth_colormap))
            else:
                detection = 0

                if imageResized:
                    disp_image = np.hstack((resized_ir_image, depth_colormap))
                else:
                    disp_image = np.hstack((ir_image, depth_colormap))

            # display_image(disp_image)

            stepCounter += 1

    finally:
        pipeline.stop()
        print('\nRealSense camera deactivated, pipeline stopped...')

# executes arucoTrack if the code is being run directly, otherwise the function can be imported
if __name__ == "__main__":
    import sys
    print('Python version:', sys.version, '\n')
    arucoTrack()