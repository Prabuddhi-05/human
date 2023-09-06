#!/usr/bin/env python3

import ros_numpy
import numpy as np
import roslib.packages
import rospy
import cv2
import pyrealsense2 # Module for Intel RealSense SDK
from sensor_msgs.msg import Image, CameraInfo # Calibration information of RealSense depth camera
from ultralytics import YOLO
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker # Visualization of 3D data in RViz
from geometry_msgs.msg import Point # Visualization of 3D points in RViz
from colorsys import hsv_to_rgb # Conversion HSV to RGB colors
from stickfig_builder import stickfig_builder # Creation of stick figure humans for RViz visualization

# Keypoints' names to their respective indices
    #NOSE:           int = 0
    #LEFT_EYE:       int = 1
    #RIGHT_EYE:      int = 2
    #LEFT_EAR:       int = 3
    #RIGHT_EAR:      int = 4
    #LEFT_SHOULDER:  int = 5
    #RIGHT_SHOULDER: int = 6
    #LEFT_ELBOW:     int = 7
    #RIGHT_ELBOW:    int = 8
    #LEFT_WRIST:     int = 9
    #RIGHT_WRIST:    int = 10
    #LEFT_HIP:       int = 11
    #RIGHT_HIP:      int = 12
    #LEFT_KNEE:      int = 13
    #RIGHT_KNEE:     int = 14
    #LEFT_ANKLE:     int = 15
    #RIGHT_ANKLE:    int = 16

marker0_time = rospy.Time()
marker1_time = rospy.Time()													
marker2_time = rospy.Time()
clear_interval = rospy.Duration(1)

person_count = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0

class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n-pose.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
        image_topic = rospy.get_param("~image_topic", "/left_camera/color/image_raw") # Raw color images captured by the left camera of the robot
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

        rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        self.image_pub = rospy.Publisher("left_debug_image", Image, queue_size=1)
        
        self.detection_pub = rospy.Publisher(detection_topic, Detection2DArray, queue_size=1)
        
        # Subscribes to camera information topic for calibration data of the left RealSense depth camera
        rospy.Subscriber("/left_camera/color/camera_info", CameraInfo, self.camerainfo_callback, queue_size=1) 
        
        # Subscribes to depth image topic for raw depth data of the left RealSense depth camera 
        rospy.Subscriber("/left_camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
               
    # Processes depth image data 
    def depth_callback(self, msg):
        # Converts the depth image message into a NumPy array
        depth_image = ros_numpy.numpify(msg) 
        
        # Resizes the depth image from 1280x720 to 640x480
        self.depth_image = cv2.resize(depth_image,(640,480))  
                     
    # Processes color image data 
    def image_callback(self, msg):
        global person_count
        global count_1
        global count_2
        global count_3
        global count_4
        status = -1
        
        header = msg.header
        encoding = msg.encoding
        numpy_image = ros_numpy.numpify(msg)
        height = numpy_image.shape[0]
        width = numpy_image.shape[1]
        results = self.model.track(
            source=numpy_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
        )
          
        # Extracts keypoints
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:  # Checks whether keypoints' information is available in the result
            # Extracts keypoints from results
            result_keypoints = results[0].keypoints.xyn.cpu().numpy()[0] 
            # Extract bounding box information from results
            bounding_boxes = results[0].boxes.xyxy.cpu().numpy() 
           
            # Iterates through each detected person's bounding boxes and keypoints
            for person_idx, bbox in enumerate(bounding_boxes):  
            
                person_count += 1 
                
                # Extracts the top-left and bottom-right corners of the bounding box for the current person
                x_min, y_min, x_max, y_max = map(int, bbox[:4]) 
            
                # Stores keypoints (x,y) for the current person inside the bounding box 
                person_keypoints = result_keypoints[: , :]
                
                # Converts normalized keypoint coordinates to pixel coordinates using image dimensions
                person_keypoints_scaled = np.transpose([np.round(person_keypoints[:,0]*width).astype(int), np.round(person_keypoints[:,1]*height).astype(int)])
                
                # Defines a dictionary that maps selected keypoints' names to their respective indices
                
                keypoints_info = {
                "nose": (0, 0),
                "left_shoulder": (5, 5),
                "right_shoulder": (6, 6),
                "left_arm": (9, 9),
                "right_arm": (10, 10),
                "left_hip": (11, 11),
                "right_hip": (12, 12),
                "left_ankle": (15, 15),
                "right_ankle": (16, 16),
                }
                
                # Creates an empty dictionary to store the 3D world coordinates of keypoints
                world_coordinates = {}
                
                # Iterates through the keypoints for each detected person and calculates their world coordinates 
                for keypoint_name, (idx_x, idx_y) in keypoints_info.items():
                    x = person_keypoints_scaled[idx_x][0] # Keypoint's x scaled pixel coordinate  
                    y = person_keypoints_scaled[idx_y][1] # Keypoint's y scaled pixel coordinate
                    depth = self.depth_image[y-1][x-1] / 1000 # Depth value of the keypoint from the depth image
                    # Using pixel coordinates, depth and camera intrinsic data to convert to world coordinates
                    forward, left, up = self.convert_depth_to_world_coordinates(x, y, depth, self.cameraInfo)
                    world_coordinates[keypoint_name] = [forward, left, up]
                    
                # Calculates the mid points between certain keypoints
                mid_shoulder = [(world_coordinates["left_shoulder"][i] + world_coordinates["right_shoulder"][i]) / 2 for i in range(3)] # Mid point between left_shoulder and right_shoulder 
                mid_hips = [(world_coordinates["left_hip"][i] + world_coordinates["right_hip"][i]) / 2 for i in range(3)] # Mid point between left_hip and right_hip
                
                # Creates a dictionary containing calculated world coordinates of different keypoints
                person_world_coordinates = {
                    "A": world_coordinates["nose"],
                    "B": mid_shoulder,
                    "C": world_coordinates["left_arm"],
                    "D": world_coordinates["right_arm"],
                    "E": mid_hips,
                    "F": world_coordinates["left_ankle"],
                    "G": world_coordinates["right_ankle"]
            	 }

                print(f"World Coordinates for Person {person_idx + 1}:")
                for keypoint, coordinates in person_world_coordinates.items():
                    print(f"{keypoint}: {coordinates}")

                # Creates an instance of 'stickfig_builder'
                self.person_instance = stickfig_builder(person_world_coordinates)

                # Calls the 'raise_baby' method to build the stick figure
                new_world_coordinates, status = self.person_instance.raise_baby()

                if status == 1:
                    count_1 += 1
                    
                elif status == 3:
                    count_3 += 1
                    
                elif status == 4:
                    count_4 += 1

                elif status == 2:
                    count_2 += 1
                          
                # Calls the 'visualize_stick_figure' method to create and publish stick figure markers representing the person's pose
                self.visualize_stick_figure(new_world_coordinates, person_idx)
                                           
        self.publish_detection(results, header)
        
        self.publish_debug_image(results, encoding)
        
        global marker0_time
        global marker1_time
        global marker2_time
        global clear_interval

        t0 = rospy.Time.now() - marker0_time
        t1 = rospy.Time.now() - marker1_time
        t2 = rospy.Time.now() - marker2_time
        
        empty_coordinates = {
                'A': [0.0, 0.0, 0.0],
                'B': [0.0, 0.0, 0.0],
                'C': [0.0, 0.0, 0.0],
                'D': [0.0, 0.0, 0.0],
                'E': [0.0, 0.0, 0.0],
                'F': [0.0, 0.0, 0.0],
                'G': [0.0, 0.0, 0.0]
                }
                
        if t0 > clear_interval:
            self.visualize_stick_figure(empty_coordinates, 0)
            
        if t1 > clear_interval:
            self.visualize_stick_figure(empty_coordinates, 1)
            
        if t2 > clear_interval:
            self.visualize_stick_figure(empty_coordinates, 2)
        
        print(status,person_count)
        if person_count == 10:
        
            # Calculate the percentages
            percentage_1 = (count_1 / person_count) * 100
            percentage_2 = (count_2 / person_count) * 100
            percentage_3 = (count_3 / person_count) * 100
            percentage_4 = (count_4 / person_count) * 100

            # Print the percentages
            print("Percentage of Case 1:", str(percentage_1))
            print("Percentage of Case 2:", str(percentage_2))
            print("Percentage of Case 3:", str(percentage_3))
            print("Percentage of Case 4:", str(percentage_4))
            
            # Define the file path where you want to save the data
            file_path = "/home/prabuddhi/catkin_ws_4/src/ultralytics_ros/counts.txt"

            # Open the file in write mode and save the percentages
            with open(file_path, 'w') as file:
                file.write(f"Percentage of Case 1: {percentage_1}\n")
                file.write(f"Percentage of Case 2: {percentage_2}\n")
                file.write(f"Percentage of Case 3: {percentage_3}\n")
                file.write(f"Percentage of Case 4: {percentage_4}\n")
            rospy.signal_shutdown("Reached Image Count Maximum")
        
    # Publishes the debug image 
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
            
    # Converts depth and pixel coordinates to world coordinates using camera intrinsics and depth data (2D-to-3D transformation)
    def convert_depth_to_world_coordinates(self, x, y, depth, cameraInfo):
        _intrinsics = pyrealsense2.intrinsics() # Creates an instance of the intrinsics class from the pyrealsense2 module
        _intrinsics.width = 640 # Width of the depth image
        _intrinsics.height = 480 # Height of the depth image
        _intrinsics.ppx = cameraInfo.K[2] # Principal point's x-coordinate.
        _intrinsics.ppy = cameraInfo.K[5] # Principal point's y-coordinate.
        _intrinsics.fx = cameraInfo.K[0] # Focal length in the x-direction
        _intrinsics.fy = cameraInfo.K[4] # Focal length in the y-direction
        _intrinsics.model = pyrealsense2.distortion.none  # No lens distortion correction is applied
        _intrinsics.coeffs = [i for i in [0.0, 0.0, 0.0, 0.0, 0.0]] # No distortion coefficients
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth) # Calculates the 3D world coordinates of the given pixel
        return result[2], -result[0], -result[1] # Forward, left, up
        
    # Stores the received camera information message    
    def camerainfo_callback(self, msg):
        self.cameraInfo = msg
        
    # Publishes detected object details including bounding boxes, classes & confidence scores 
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
            
    # Generates a stick figure marker for visualization purposes       
    def visualize_stick_figure(self, new_world_coordinates, person_idx, marker_topic='stick_figure_marker_'):
        
        color = self.get_person_color(person_idx) # Retrieves the color for a specific person
        
        topic = marker_topic + str(person_idx)
       
        marker = Marker() # Creates a new Marker instance
        marker.header.frame_id = 'left_camera_link'  # Sets the frame ID for the marker's reference frame (coordinate system)
        marker.type = Marker.LINE_LIST # Sets the type of marker to be drawn (LINE)
        marker.action = Marker.ADD # Sets the action to be taken on the marker (Add a new marker)
        marker.scale.x = 0.05  # Sets the scale of the marker's lines
        marker.color.a = 1.0 # Sets the alpha (transparency) value for the marker's color
        #marker.color.r = 1.0  # Set the red component of the marker's color to 1.0 (full intensity red)
        
        if marker_topic == 'stick_figure_marker_':
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
        
        # Specifies links between keypoints for stick figure 
        
        keypoint_links = [
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("B", "E"),
        ("E", "F"),
        ("E", "G")
        ]
        
        # Fills the marker points with links between keypoints' 3D coordinates
        for start_key, end_key in keypoint_links:
            start_point = new_world_coordinates[start_key]
            end_point = new_world_coordinates[end_key]
          
            # Creates a Point instance for the starting point of the link
            p1 = Point()
            p1.x = start_point[0]
            p1.y = start_point[1]
            p1.z = start_point[2]
            
            # Creates a Point instance for the ending point of the link
            p2 = Point()
            p2.x = end_point[0]
            p2.y = end_point[1]
            p2.z = end_point[2]
            
            # Adds the starting and ending points to the marker's list of points
            marker.points.append(p1)
            marker.points.append(p2)
            
        # Creates a publisher for the stick figure marker
        marker_pub = rospy.Publisher(topic, Marker, queue_size=1)
        
        if person_idx == 0:
            global marker0_time
            marker0_time = rospy.Time.now()
            
        elif person_idx == 1:
            global marker1_time
            marker1_time = rospy.Time.now()
            
        elif person_idx == 2:
            global marker2_time
            marker2_time = rospy.Time.now()
        
        # Publishes the marker to the 'stick_figure_marker' topic
        marker_pub.publish(marker)
        
        # Delays for a short time (0.1 seconds) to allow the marker to be processed and visualized
        rospy.sleep(0.1)
        
    # Generates distinct colors for each person based on their index  
    def get_person_color(self, person_idx):
        # Calculates a hue value using the golden ratio formula
        hue = (person_idx * 0.618033988749895) % 1.0  

        # Converts the HSV color to RGB
        color_rgb = hsv_to_rgb(hue, 1.0, 1.0)

        return color_rgb
        
if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin() 
