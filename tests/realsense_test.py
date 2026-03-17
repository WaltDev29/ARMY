import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline() # Create a pipeline

config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start() # Start streaming

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        frame = np.asanyarray(depth_frame.get_data())
        cv2.imshow("frame", frame)

        width, height = depth_frame.get_width(), depth_frame.get_height()
        dist = depth_frame.get_distance(width // 2, height // 2)
        print(f"The camera is facing an object {dist:.3f} meters away", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        

finally:
    pipeline.stop() # Stop streaming