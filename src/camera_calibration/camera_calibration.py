#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageCoordinates:

    def __init__(self):
        rospy.loginfo("Running image coordinates node...")
        self.cv_image = None
        self.resolution = (960, 528)
        self.topic_name = "/camera/image"
        self.rate = rospy.Rate(30)
        self.coordinates = []
        rospy.loginfo(f"Subscribing to {self.topic_name}")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            self.topic_name, Image, self.image_callback)
        
        self.image_pub = rospy.Publisher(
            "/image_raw_bin", Image, queue_size=10)
        rospy.loginfo("Image coordinates node ready")

    def image_callback(self, data):
        """
        Callback function for image subscriber
        """
        # Convert image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # Resize image
        cv_image = cv2.resize(cv_image, self.resolution)
        # Convert to grayscale
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Save image
        self.cv_image = cv_image        
        # Show image
        cv2.imshow("Image", cv_image)
        key = cv2.waitKey(1)
        if key == 32:
            self.take_screenshot()

    def display_image(self):
        """
        Display the last received image
        """
        if self.cv_image is not None:
            cv2.imshow("Image", self.cv_image)
            cv2.setMouseCallback("Image", self.mouse_callback)
            cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback function for mouse events
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            rospy.loginfo(f"Clicked coordinates: ({x}, {y})")
            x_w = input("Enter world x coordinate: ")
            y_w = input("Enter world y coordinate: ")
            z_w = input("Enter world z coordinate: ")
            self.coordinates.append(([x, y], [x_w, y_w, z_w]))
            rospy.loginfo(f"World coordinates: ({x_w}, {y_w}, {z_w})")
            rospy.loginfo("-"*50)


    def take_screenshot(self):
        """
        Take a screenshot of the current image
        """
        rospy.loginfo("Screenshot taken")
        # Unsubscribe from image topic
        self.image_sub.unregister()
        while not rospy.is_shutdown():
            self.display_image()
            self.rate.sleep()



def main():
    rospy.init_node('image_coordinates')
    image_coordinates = ImageCoordinates()
    rospy.spin()


if __name__ == '__main__':
    main()