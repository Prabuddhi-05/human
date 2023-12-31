#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImageCombiner:
    def __init__(self):
        # Initializes CvBridge for converting between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        
        # Dictionary to store received images with their associated positions
        self.images = {}  
        
        # Specifies the order in which images will be combined
        self.image_order = ["front", "right", "back", "left"]

        # Subscribes to each debug image topic
        for topic in self.image_order:
            sub = rospy.Subscriber("debug_image_" + topic, Image, self.image_callback, callback_args=topic, queue_size=1)
            self.images[topic] = None
            
        # Publisher for the combined image
        self.image_combined_pub = rospy.Publisher("debug_image_combined", Image, queue_size=1)
       
    def image_callback(self, msg, topic):
        # Converts ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Converts RGB images to BGR format for concatenation
        if cv_image.shape[-1] == 4:
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        else:
            cv_image_bgr = cv_image
            
        # Stores the received image in the dictionary based on its position
        self.images[topic] = cv_image_bgr
        
        # Publishes the combined image when all images are received
        self.publish_combined_image()

    def publish_combined_image(self):
    
        # Gets the images in the specified order, filter out any None values (unreceived images)
        combined_images = [self.images[topic] for topic in self.image_order if self.images[topic] is not None]
        
        # Check if all expected images are received
        if len(combined_images) == len(self.image_order):
        
            # Concatenates images along the horizontal axis
            combined_row = np.concatenate(combined_images, axis=1)

            # Converts the combined image to BGR format for display
            combined_row_bgr = cv2.cvtColor(combined_row, cv2.COLOR_RGB2BGR)

            # Converts the combined image to ROS message format
            combined_image_msg = self.bridge.cv2_to_imgmsg(combined_row_bgr, encoding="bgr8")

            # Publishes the combined image
            self.image_combined_pub.publish(combined_image_msg)

if __name__ == "__main__":
    rospy.init_node("image_combiner")
    combiner = ImageCombiner()
    rospy.spin()

