#!/usr/bin/env python3

import ros_numpy
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge


class TrackerNode:
    def __init__(self, image_topic, debug_image_topic):
        # Store the received image topic and debug image topic
        self.image_topic = image_topic
        self.debug_image_topic = debug_image_topic

        yolo_model = rospy.get_param("~yolo_model", "yolov8n-pose.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
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
        
        # Subscriber initialization
        self.subscriber = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

        # Publisher initialization for detection results
        self.detection_pub = rospy.Publisher(
            detection_topic, Detection2DArray, queue_size=1
        )

        # Publisher initialization for debug image
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        # Initialize image storage dictionary
        self.image_dict = {}
        
        # Initialize CvBridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Store the received images in the dictionary with their positions
        numpy_image = ros_numpy.numpify(msg)
        
        # Define the desired width and height for the resized images
        desired_width = 128
        desired_height = 96
    
        # Resize the received image using OpenCV
        resized_image = cv2.resize(numpy_image, (desired_width, desired_height))
        
        # Perform object detection on the each image
        results = self.model.track(
            source=resized_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
        )
        
        #torch.cuda.empty_cache()

        # Publish the detection results and the debug image
        self.publish_detection(results, msg.header)
            
        # Plot debug image and publish to debug image topic
        debug_image = None
        if self.debug and results is not None:
            debug_image = results[0].plot(
                conf=self.debug_conf,
                line_width=self.debug_line_width,
                font_size=self.debug_font_size,
                font=self.debug_font,
                labels=self.debug_labels,
                boxes=self.debug_boxes,
            )

        if debug_image is not None:
            if debug_image.shape[-1] == 3:
            # Convert debug_image to BGR format for display
                debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
                debug_image_msg = self.bridge.cv2_to_imgmsg(debug_image_bgr, encoding="bgr8")
                self.debug_image_pub.publish(debug_image_msg)
            else:
                rospy.logwarn("Invalid debug_image format. Unable to convert for display.")
            # Inside the image_callback method

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
    #torch.backends.cuda.max_split_size_mb = 512  # Set this value to a suitable amount
    rospy.init_node("tracker_node")
    
    front_camera_image_topic = rospy.get_param("~front_camera_image_topic", "/front_camera/color/image_raw")
    debug_image_front_topic = "debug_image_front"  
    node_front = TrackerNode(image_topic=front_camera_image_topic, debug_image_topic=debug_image_front_topic)
    
    right_camera_image_topic = rospy.get_param("~right_camera_image_topic", "/right_camera/color/image_raw")
    debug_image_right_topic = "debug_image_right"  
    node_right = TrackerNode(image_topic=right_camera_image_topic, debug_image_topic=debug_image_right_topic)
    
    back_camera_image_topic = rospy.get_param("~back_camera_image_topic", "/back_camera/color/image_raw")
    debug_image_back_topic = "debug_image_back"  
    node_back = TrackerNode(image_topic=back_camera_image_topic, debug_image_topic=debug_image_back_topic)
    
    left_camera_image_topic = rospy.get_param("~left_camera_image_topic", "/left_camera/color/image_raw")
    debug_image_left_topic = "debug_image_left" 
    node_left = TrackerNode(image_topic=left_camera_image_topic, debug_image_topic=debug_image_left_topic)
    
    rospy.spin()
