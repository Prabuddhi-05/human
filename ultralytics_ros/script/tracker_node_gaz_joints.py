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
import os
from cv_bridge import CvBridge
from datetime import datetime
from pydantic import BaseModel

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class TrackerNode:
    def __init__(self):
        # Read the image_topics parameter as a comma-separated string and convert to a list
        image_topics = rospy.get_param("~image_topics", "").split(',')

        # Map image topics to their corresponding positions (front, right, back, left)
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
        self.debug_image_width = rospy.get_param("~debug_image_width",120)
        self.debug_image_height = rospy.get_param("~debug_image_height",640)

        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")

        # Subscribe to the four image topics
        self.subscribers = []
        for topic in image_topics:
            sub = rospy.Subscriber(
                topic, Image, self.image_callback, callback_args=topic, queue_size=1, buff_size=2**24
            )
            self.subscribers.append(sub)

        # Publisher initialization
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(
            detection_topic, Detection2DArray, queue_size=1
        )

        # Initialize image storage dictionary
        self.image_dict = {}

        # Add a member variable for the image saving folder
        self.image_save_folder = "/home/prabuddhi/catkin_ws_4/src/ultralytics_ros/script/imgs"
        
        # Create an instance of the Keypoint class
        self.get_keypoint = GetKeypoint()

    def image_callback(self, msg, topic):
        # Store the received images in the dictionary with their positions
        numpy_image = ros_numpy.numpify(msg)
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

            # Combine the four images into a single row image (tile them together)
            combined_image = np.hstack(sorted_images)

            # Clear the image dictionary for the next set of four
            self.image_dict = {}

            # Perform object detection on the combined image
            results = self.model.track(
                source=combined_image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                classes=self.classes,
                tracker=self.tracker,
                verbose=False,
            )

            # Extract keypoints and print to the terminal
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                result_keypoints = results[0].keypoints.xyn.cpu().numpy()[0]
                #rospy.loginfo("Detected Keypoints: %s", result_keypoints)
                
                # Iterate through each joint index and extract coordinates
                for joint_index in range(17):
                    joint_x, joint_y = result_keypoints[joint_index]
                    joint_name = self.get_joint_name(joint_index)  # A helper function to get joint name
                    rospy.loginfo("%s: (%f, %f)", joint_name, joint_x, joint_y)
                    rospy.loginfo("---------------------------------")
                
            # Save the original image to the image saving folder
            cv_image = ros_numpy.numpify(msg)
            image_filename = f"{rospy.Time.now().to_sec()}.jpg"
            image_path = os.path.join(self.image_save_folder, image_filename)
            cv2.imwrite(image_path, cv_image)

            # Publish the detection results and the debug image
            self.publish_detection(results, msg.header)
            self.publish_debug_image(results, msg.encoding)
            
    def get_joint_name(self, joint_index):
        # Helper function to get the name of a joint based on its index
        joint_names = [
            "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
        ]
        return joint_names[joint_index]

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

