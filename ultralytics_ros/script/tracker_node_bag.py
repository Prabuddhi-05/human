#!/usr/bin/env python3

import numpy as np
import cv2
import roslib.packages
import rospy
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import time
import pynvml  # Imports the pynvml library for GPU memory monitoring

class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n-pose.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
        image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_color/compressed")  # Images in the compressed format  
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.debug = rospy.get_param("~debug", True)
        self.debug_conf = rospy.get_param("~debug_conf", True)
        self.debug_line_width = rospy.get_param("~debug_line_width", 1)
        self.debug_font_size = rospy.get_param("~debug_font_size", 1)
        self.debug_font = rospy.get_param("~debug_font", "Arial.ttf")
        self.debug_labels = rospy.get_param("~debug_labels", True)
        self.debug_boxes = rospy.get_param("~debug_boxes", True)
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.sub = rospy.Subscriber(
            image_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(
            detection_topic, Detection2DArray, queue_size=1
        )
        self.bridge = CvBridge()
        
        # Initializes NVML for GPU memory monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

    def image_callback(self, msg):
    
        # Records the starting time
        start_time = time.time()
        # Decompress the compressed image data
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        header = msg.header
        encoding = msg.format
        
        # Resize the image to the desired dimensions
        desired_width = 640
        desired_height = 480
        cv_image = cv2.resize(cv_image, (desired_width, desired_height))

        # Tile the image 4 times horizontally
        tiled_image = np.hstack([cv_image, cv_image, cv_image, cv_image])

        # Process the tiled image using the YOLO model
        results = self.model.track(
            source=tiled_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
        )
        
        # Records the ending time
        end_time = time.time()
        
        # Calculates the inference time
        inference_time = end_time - start_time

        # Prints the inference time
        print("Inference Time: %.4f seconds" % inference_time)
        
        # Calculates FPS
        if inference_time > 0:
            fps = 1.0 / inference_time
        else:
            fps = 0.0
                
        print("FPS: %.2f" % fps)  # Prints the FPS value
        
        # Monitors GPU memory usage
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        used_memory_mb = gpu_info.used / (1024 ** 2)  
        print(f"GPU Memory Used: {used_memory_mb:.2f} GB")

        # Publish the detection results
        self.publish_detection(results, header)

        # Convert the image back to sensor_msgs/Image and publish for debugging
        if self.debug and results is not None:
            debug_image_msg = self.bridge.cv2_to_imgmsg(results[0].plot(
                conf=self.debug_conf,
                line_width=self.debug_line_width,
                font_size=self.debug_font_size,
                font=self.debug_font,
                labels=self.debug_labels,
                boxes=self.debug_boxes,
            ), encoding="bgr8")
            self.image_pub.publish(debug_image_msg)

    def publish_detection(self, results, header):
        if results is not None:
            detections_msg = Detection2DArray()
            detections_msg.header = header
            bounding_box = results[0].boxes.xywh
            classes = results[0].boxes.cls
            confidence_score = results[0].boxes.conf
            for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                detection = Detection2D()
                detection.bbox.center.x = float(bbox[0])
                detection.bbox.center.y = float(bbox[1])
                detection.bbox.size_x = float(bbox[2])
                detection.bbox.size_y = float(bbox[3])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(cls)
                hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)
            self.detection_pub.publish(detections_msg)


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()

