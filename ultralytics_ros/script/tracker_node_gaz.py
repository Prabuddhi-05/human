#!/usr/bin/env python3

import ros_numpy
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (Detection2D, Detection2DArray,ObjectHypothesisWithPose)
import cv2
import numpy as np
import time
import pynvml  # Imports the pynvml library for GPU memory monitoring

image_count = 0

class TrackerNode:
    def __init__(self):
    
        # Reads the image_topics parameter as a comma-separated string and convert to a list
        image_topics = rospy.get_param("~image_topics", "").split(',')

        # Maps image topics to their corresponding positions (front, right, back, left)
        self.image_topic_to_position = {
            image_topics[0]: "front",
            image_topics[1]: "right",
            image_topics[2]: "back",
            image_topics[3]: "left",
        }

        yolo_model = rospy.get_param("~yolo_model", "yolov8n-pose.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
        self.conf_thres = rospy.get_param("~conf_thres", 0.08)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.debug = rospy.get_param("~debug", False)
        self.debug_conf = rospy.get_param("~debug_conf", True)
        self.debug_line_width = rospy.get_param("~debug_line_width", None)
        self.debug_font_size = rospy.get_param("~debug_font_size", None)
        self.debug_font = rospy.get_param("~debug_font", "Arial.ttf")
        self.debug_labels = rospy.get_param("~debug_labels", True)
        self.debug_boxes = rospy.get_param("~debug_boxes", True)

        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")

        # Subscribes to the four image topics
        self.subscribers = []
        for topic in image_topics:
            sub = rospy.Subscriber(topic, Image, self.image_callback, callback_args=topic, queue_size=1, buff_size=2**24)
            self.subscribers.append(sub)  # List used to store the subscriber objects for each image topic 
        
        # Publisher initialization
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(detection_topic, Detection2DArray, queue_size=1)

        # Initializes image storage dictionary
        self.image_dict = {}
        
        # Initializes NVML for GPU memory monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
        
        self.image_count = 0  # Initializes image_count

    def image_callback(self, msg, topic):
    
        global image_count
        image_count += 1
    
        # Records the starting time
        start_time = time.time()
        
        # Converts the received ROS image message (msg) to a NumPy array
        numpy_image = ros_numpy.numpify(msg)
        
        # Stores the NumPy image in the image_dict dictionary with its position as the key
        # The 'self.image_topic_to_position' dictionary maps the topic to its corresponding position
        self.image_dict[self.image_topic_to_position[topic]] = numpy_image

        # When all four images are received, proceed with processing
        if len(self.image_dict) == 4:
            # Sort the images based on their positions (front, right, back, left)
            sorted_images = [
                self.image_dict["front"],
                self.image_dict["right"],
                self.image_dict["back"],
                self.image_dict["left"]
            ]

            # Combines the four images into a single row image (tile them together)
            combined_image = np.hstack(sorted_images)

            # Clears the image dictionary for the next set of four
            self.image_dict = {}

            # Performs object detection on the combined image
            results = self.model.track(
                source=combined_image,
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
   
            # Monitors GPU memory usage
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_memory_mb = gpu_info.used / (1024 ** 2)  
            print(f"GPU Memory Used: {used_memory_mb:.2f} MB")

            # Publishes the detection results and the debug image
            self.publish_detection(results, msg.header)
            self.publish_debug_image(results, msg.encoding)
           
        
    def publish_debug_image(self, results, encoding):
        if self.debug and results is not None:
            plotted_image = results[0].plot(
                conf=self.debug_conf,
                line_width=self.debug_line_width,
                font_size=self.debug_font_size,
                font=self.debug_font,
                labels=self.debug_labels,
                boxes=self.debug_boxes,
            )
         
            debug_image_msg = ros_numpy.msgify(Image, plotted_image, encoding=encoding)
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

