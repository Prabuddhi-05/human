#!/usr/bin/env python3

# Imports necessary ROS libraries and messages
import ros_numpy
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)
import cv2
import time
import pynvml  # Imports the pynvml library for GPU memory monitoring

# Defines a class for the ROS node
class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n-pose.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw") # Raw color images from the RealSense camera 
        #image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw") # Raw color images from the Web camera 
        self.conf_thres = rospy.get_param("~conf_thres", 0.25) # Confidence threshold 
        self.iou_thres = rospy.get_param("~iou_thres", 0.45) # Intersection over Union (IoU) 
        self.max_det = rospy.get_param("~max_det", 300) # Maximum number of detections
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml") # Tracker
        self.debug = rospy.get_param("~debug", True)
        self.debug_conf = rospy.get_param("~debug_conf", True)
        self.debug_line_width = rospy.get_param("~debug_line_width", 1)
        self.debug_font_size = rospy.get_param("~debug_font_size", 1)
        self.debug_font = rospy.get_param("~debug_font", "Arial.ttf")
        self.debug_labels = rospy.get_param("~debug_labels", True)
        self.debug_boxes = rospy.get_param("~debug_boxes", True)
        
        # Gets the package path and initializes the YOLO model
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        
        # Subscribes to the ROS image topic and sets up publishers
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(detection_topic, Detection2DArray, queue_size=1)
        
        # Initializes NVML for GPU memory monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
        
        self.total_fps = 0.0
        self.total_inference_time = 0.0
        self.total_gpu_memory = 0.0

    # Callback function for processing images
    def image_callback(self, msg):
    
        # Records the starting time
        start_time = time.time()
        
        header = msg.header
        encoding = msg.encoding
        numpy_image = ros_numpy.numpify(msg)
        
        # Uses the YOLO model for object tracking
        results = self.model.track(
            source=numpy_image,
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
        
        # Accumulate values for averaging
        self.total_fps += fps
        self.total_inference_time += inference_time
        
        # Monitors GPU memory usage
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        used_memory_mb = gpu_info.used / (1024 ** 2)  # Converts to megabytes
        print(f"GPU Memory Used: {used_memory_mb:.2f} MB")
        #self.total_gpu_memory += used_memory_mb
        
        # Publishes detection results and debug image
        self.publish_detection(results, header)
        self.publish_debug_image(results, encoding,fps)
        
    # Publishes a debug image with bounding boxes and labels
    def publish_debug_image(self, results, encoding,fps):
        if self.debug and results is not None:
            plotted_image = results[0].plot(
                conf=self.debug_conf,
                line_width=self.debug_line_width,
                font_size=self.debug_font_size,
                font=self.debug_font,
                labels=self.debug_labels,
                boxes=self.debug_boxes,
            )
            
            # Add FPS text to the image
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(
                plotted_image,
                fps_text,
                (10, 30),  # Position of the text on the image
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Red color for the text (BGR)
                2,  # Thickness of the text
                cv2.LINE_AA,
            )
            
            debug_image_msg = ros_numpy.msgify(Image, plotted_image, encoding=encoding)
            self.image_pub.publish(debug_image_msg)
            
    # Publishes detection results as Detection2DArray messages
    def publish_detection(self, results, header):
        if results is not None:
            detections_msg = Detection2DArray()
            detections_msg.header = header
            bounding_box = results[0].boxes.xywh
            classes = results[0].boxes.cls
            confidence_score = results[0].boxes.conf
            # Creates Detection2D messages for each detected object
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
                
            # Publishes the Detection2DArray message
            self.detection_pub.publish(detections_msg)

# Main execution block
if __name__ == "__main__":
    # Initializes the ROS node
    rospy.init_node("tracker_node")
    # Creates an instance of the TrackerNode class
    node = TrackerNode()
    # Enters the ROS spin loop to keep the node running
    rospy.spin()
