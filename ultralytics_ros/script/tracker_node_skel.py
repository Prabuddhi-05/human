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
        self.debug = rospy.get_param("~debug", False)
        self.debug_conf = rospy.get_param("~debug_conf", True)
        self.debug_line_width = rospy.get_param("~debug_line_width", None)
        self.debug_font_size = rospy.get_param("~debug_font_size", None)
        self.debug_font = rospy.get_param("~debug_font", "Arial.ttf")
        self.debug_labels = rospy.get_param("~debug_labels", True)
        self.debug_boxes = rospy.get_param("~debug_boxes", True)
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        rospy.Subscriber(
            image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(
            detection_topic, Detection2DArray, queue_size=1
        )
        
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
                    depth = self.depth_image[y][x] / 1000 # Depth value of the keypoint from the depth image
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

                #print(f"World Coordinates for Person {person_idx + 1}:")
                #for keypoint, coordinates in person_world_coordinates.items():
                    #print(f"{keypoint}: {coordinates}")
                          
                # Calls the 'visualize_stick_figure' method to create and publish stick figure markers representing the person's pose
                self.visualize_stick_figure()
                                           
        self.publish_detection(results, header)
        
        self.publish_debug_image(results, encoding)  
                           

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
    def visualize_stick_figure(self):
       
        # Creates a publisher for the stick figure marker 
        marker_pub = rospy.Publisher(f'stick_figure_marker', Marker, queue_size=1)
        
        # Creates a new Marker instance
        marker = Marker() # Creates a new Marker instance
        marker.header.frame_id = 'left_camera_link'  # Sets the frame ID for the marker's reference frame (coordinate system)
        marker.type = Marker.LINE_LIST # Sets the type of marker to be drawn (LINE)
        marker.action = Marker.ADD # Sets the action to be taken on the marker (Add a new marker)
        marker.scale.x = 0.05  # Sets the scale of the marker's lines
        marker.color.a = 1.0 # Sets the alpha (transparency) value for the marker's color
        marker.color.r = 1.0  # Set the red component of the marker's color to 1.0 (full intensity red)

        #world_coordinates = {
            #"nose": [5.10450005531311, 0.25099746137857437, 1.4801827669143677],
            #"mid_shoulder": [5.10450005531311, 0.25099746137857437, 1.39897882938385],
            #"left_arm": [4.945000171661377, 0.17762337625026703, 1.0870550870895386],
            #"right_arm":  [5.323999881744385, 0.35187602043151855, 1.0870550870895386],
            #"mid_hips":  [5.10450005531311, 0.25099746137857437, 1.0374685525894165],
            #"left_ankle": [4.879000186920166, 0.14721225202083588, 0.5257580280303955],
            #"right_ankle":  [5.265999794006348, 0.3253442347049713, 0.5257580280303955]
        #}
        
        # World coordinates of the perfect skeleton
        world_coordinates = {
        "nose": [6.510999917984009, -3.154663920402527, 1.0743868350982666],
        "mid_shoulder": [6.510999917984009, -3.154663920402527, 0.8816350102424622],
        "left_arm": [5.927999973297119, -3.1555252075195312, 0.21806474030017853],
        "right_arm":  [7.041999816894531, -3.1555252075195312, 0.28951960802078247],
        "mid_hips":  [6.510999917984009, -3.154663920402527, 0.3209571838378906],
        "left_ankle": [5.927999973297119, -3.1555252075195312, -0.6402308344841003],
        "right_ankle":  [7.041999816894531, -3.1555252075195312, -0.6400945782661438]
        }
          
        # Specifies links between keypoints for stick figure 
        keypoint_links = [
        ("nose", "mid_shoulder"),
        ("mid_shoulder", "left_arm"),
        ("mid_shoulder", "right_arm"),
        ("mid_shoulder", "mid_hips"),
        ("mid_hips", "left_ankle"),
        ("mid_hips", "right_ankle")
        ]
        
        total_distance = 0.0

        # Fills the marker points with links between keypoints' 3D coordinates
        for start_key, end_key in keypoint_links:
            start_point = world_coordinates[start_key]
            end_point = world_coordinates[end_key]
            
            # Calculates Euclidean distance between the two points
            distance = np.linalg.norm(np.array(start_point) - np.array(end_point))
            
            # Calculates the midpoint between arms
            mid_point_arms = [(world_coordinates["left_arm"][i] + world_coordinates["right_arm"][i]) / 2 for i in range(3)]
            
            # Calculates the midpoint between ankles
            mid_point_ankles = [(world_coordinates["left_ankle"][i] + world_coordinates["right_ankle"][i]) / 2 for i in range(3)]
            
            # Calculates the Euclidean distance between nose and mid_shoulder
            distance_nose_mid_shoulder = np.linalg.norm(np.array(world_coordinates["nose"]) - np.array(world_coordinates["mid_shoulder"]))
            # Calculates the Euclidean distance between mid_shoulder and mid_hips
            distance_mid_shoulder_mid_hips = np.linalg.norm(np.array(world_coordinates["mid_shoulder"]) - np.array(world_coordinates["mid_hips"]))
            # Calculates the ratio
            ratio_nose_mid_hips = distance_nose_mid_shoulder / distance_mid_shoulder_mid_hips
            
            # Calculates the Euclidean distance between mid_shoulder and left_arm
            distance_mid_shoulder_left_arm = np.linalg.norm(np.array(world_coordinates["mid_shoulder"]) - np.array(world_coordinates["left_arm"]))
            
            # Calculates the Euclidean distance between left_arm and right_arm
            distance_left_arm_right_arm = np.linalg.norm(np.array(world_coordinates["left_arm"]) - np.array(world_coordinates["right_arm"]))
            
            # Calculates the Euclidean distance between mid_shoulder and mid_point_arms
            distance_mid_shoulder_mid_arms = np.sqrt((distance_mid_shoulder_left_arm)**2 - (distance_left_arm_right_arm/2)**2)
            
            # Calculates the Euclidean distance between mid_hips and left_ankle
            distance_mid_hips_left_ankle = np.linalg.norm(np.array(world_coordinates["mid_hips"]) - np.array(world_coordinates["left_ankle"]))
            
            # Calculates the Euclidean distance between left_ankle and right_ankle
            distance_left_ankle_right_ankle = np.linalg.norm(np.array(world_coordinates["left_ankle"]) - np.array(world_coordinates["right_ankle"]))
            
            # Calculates the Euclidean distance between mid_hips and mid_point_ankles
            distance_mid_hips_mid_ankles = np.sqrt((distance_mid_hips_left_ankle)**2 - (distance_left_ankle_right_ankle/2)**2)
            
            # New coordinates of left_arm
            
            # Calculates the direction vector of the line formed by mid_shoulder and mid_hips
            direction_vector = np.array(world_coordinates["mid_hips"]) - np.array(world_coordinates["mid_shoulder"])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)

            # Calculates the midpoint of the line segment
            midpoint = (np.array(world_coordinates["mid_shoulder"]) + np.array(world_coordinates["mid_hips"])) / 2

            # Calculates the vector between right_arm and the midpoint
            vector_to_midpoint = midpoint - np.array(world_coordinates["right_arm"])

            # Mirrors the vector around the direction vector
            mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

            # Calculates the new left_arm coordinates
            new_left_arm = midpoint + mirrored_vector
            
            # New coordinates of left_ankle
            
            # Calculates the direction vector of the line formed by mid_shoulder and mid_hips
            direction_vector = np.array(world_coordinates["mid_hips"]) - np.array(world_coordinates["mid_shoulder"])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)

            # Calculates the midpoint of the line segment
            midpoint = (np.array(world_coordinates["mid_shoulder"]) + np.array(world_coordinates["mid_hips"])) / 2

            # Calculates the vector between right_ankle and the midpoint
            vector_to_midpoint = midpoint - np.array(world_coordinates["right_ankle"])

            # Mirrors the vector around the direction vector
            mirrored_vector = vector_to_midpoint - 2 * (np.dot(vector_to_midpoint, direction_vector)) * direction_vector

            # Calculates the new left_ankle coordinates
            new_left_ankle = midpoint + mirrored_vector

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
            
            print(f"Start Key: {start_key}, Start Point: {start_point}")
            print(f"End Key: {end_key}, End Point: {end_point}")
            print(f"Distance between {start_key} and {end_key}: {distance} units")
        print(f"Ratio between nose and mid_hips: {ratio_nose_mid_hips}")
        #print(f"Distance between left_arm and right_arm: {distance_left_arm_right_arm}")
        print("Midpoint between left_arm and right_arm:", mid_point_arms)
        print(f"Distance between mid_shoulder and mid_arms: {distance_mid_shoulder_mid_arms}")
        print("Midpoint between left_ankle and right_ankle:", mid_point_ankles)
        #print(f"Distance between left_ankle and right_ankle: {distance_left_ankle_right_ankle}")
        print(f"Distance between mid_hips and mid_ankles: { distance_mid_hips_mid_ankles}")
        # Print the new left_arm coordinates
        print("New left_arm coordinates:", new_left_arm)
        print("New left_ankle coordinates:", new_left_ankle)
        print("-----------------------------------------------------------------")
               
        # Publishes the marker to the 'stick_figure_marker' topic
        marker_pub.publish(marker)
        
        # Delays for a short time (0.1 seconds) to allow the marker to be processed and visualized
        rospy.sleep(0.1)
        
if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
    
