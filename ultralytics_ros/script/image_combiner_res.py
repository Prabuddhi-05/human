#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImageCombiner:
    def __init__(self):
        self.bridge = CvBridge()
        self.images = {}  # Store received images
        self.image_order = ["front", "right", "back", "left"]

        # Subscribe to each debug image topic
        for topic in self.image_order:
            sub = rospy.Subscriber(
                "debug_image_" + topic, Image, self.image_callback, callback_args=topic, queue_size=1
            )
            self.images[topic] = None

        self.image_combined_pub = rospy.Publisher("debug_image_combined", Image, queue_size=1)

    def image_callback(self, msg, topic):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Convert RGB images to BGR format for concatenation
        if cv_image.shape[-1] == 3:
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        else:
            cv_image_bgr = cv_image

        self.images[topic] = cv_image_bgr
        self.publish_combined_image()

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

            # For debugging: save the combined image locally 
            #cv2.imwrite("combined_image.png", combined_row_bgr)  # Save image as PNG

if __name__ == "__main__":
    rospy.init_node("image_combiner")
    combiner = ImageCombiner()
    rospy.spin()

