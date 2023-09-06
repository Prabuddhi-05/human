#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import pynvml
import time
from collections import defaultdict

class TrackerNode:
    def __init__(self, image_topic, debug_image_topic):
        self.bridge = CvBridge()
        self.images = {}  # Store received images
        self.image_order = ["front", "right", "left"]  # Update this list with your camera topics
        self.gpu_handle = None

        # Subscribe to each debug image topic
        for topic in self.image_order:
            sub = rospy.Subscriber(
                "debug_image_" + topic, Image, self.image_callback, callback_args=topic, queue_size=1
            )
            self.images[topic] = None

        self.image_combined_pub = rospy.Publisher("debug_image_combined", Image, queue_size=1)

        # Initialize variables to track metrics for each image topic
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.frame_count = defaultdict(int)

        # Initializes NVML for GPU memory monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def image_callback(self, msg, topic):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Convert RGB images to BGR format for concatenation
        if cv_image.shape[-1] == 3:
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        else:
            cv_image_bgr = cv_image

        self.images[topic] = cv_image_bgr
        self.publish_combined_image()

        # Calculate inference time and FPS
        if self.metrics[topic]["frame_count"] < 100:
            start_time = time.time()

            # Simulate object detection (replace with your actual detection code)
            # For demonstration, we sleep for a short time.
            time.sleep(0.1)

            end_time = time.time()
            inference_time = end_time - start_time
            self.metrics[topic]["inference_time"] += inference_time
            self.metrics[topic]["frame_count"] += 1

            if inference_time > 0:
                fps = 1.0 / inference_time
                self.metrics[topic]["fps"] += fps

            # Monitors GPU memory usage
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_memory_mb = gpu_info.used / (1024 ** 2)  # Converts to megabytes
            self.metrics[topic]["gpu_memory"] += used_memory_mb

            if self.metrics[topic]["frame_count"] == 100:
                # Calculate and print average values after processing 100 frames
                average_fps = self.metrics[topic]["fps"] / self.metrics[topic]["frame_count"]
                average_inference_time = self.metrics[topic]["inference_time"] / self.metrics[topic]["frame_count"]
                average_gpu_memory = self.metrics[topic]["gpu_memory"] / self.metrics[topic]["frame_count"]

                print(f"Image Topic '{topic}':")
                print(f"Average FPS: {average_fps:.2f}")
                print(f"Average Inference Time: {average_inference_time:.4f} seconds")
                print(f"Average GPU Memory Used: {average_gpu_memory:.2f} MB")

                # Reset metrics for the next 100 frames
                self.metrics[topic] = defaultdict(float)
                self.frame_count[topic] = 0

    def publish_combined_image(self):
        combined_images = [self.images[topic] for topic in self.image_order if self.images[topic] is not None]
        if len(combined_images) == len(self.image_order):
            # Concatenate images along the horizontal axis
            combined_row = np.concatenate(combined_images, axis=1)

            # Convert the combined image to BGR format for display
            combined_row_bgr = cv2.cvtColor(combined_row, cv2.COLOR_RGB2BGR)

            # Convert the combined image to ROS message format
            combined_image_msg = self.bridge.cv2_to_imgmsg(combined_row_bgr, encoding="bgr8")

            # Publish the combined image
            self.image_combined_pub.publish(combined_image_msg)

if __name__ == "__main__":
    rospy.init_node("tracker_node")

    # Initialize the tracker node for each camera topic
    front_camera_image_topic = rospy.get_param("~front_camera_image_topic", "/front_camera/color/image_raw")
    debug_image_front_topic = "debug_image_front"
    node_front = TrackerNode(image_topic=front_camera_image_topic, debug_image_topic=debug_image_front_topic)

    right_camera_image_topic = rospy.get_param("~right_camera_image_topic", "/right_camera/color/image_raw")
    debug_image_right_topic = "debug_image_right"
    node_right = TrackerNode(image_topic=right_camera_image_topic, debug_image_topic=debug_image_right_topic)

    left_camera_image_topic = rospy.get_param("~left_camera_image_topic", "/left_camera/color/image_raw")
    debug_image_left_topic = "debug_image_left"
    node_left = TrackerNode(image_topic=left_camera_image_topic, debug_image_topic=debug_image_left_topic)

    # Spin the ROS node
    rospy.spin()

