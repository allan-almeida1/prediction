#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageCoordinates:

    def __init__(self):
        rospy.loginfo("Running image coordinates node...")
        self.resolution = (960, 528)
        self.topic_name = "/camera/image"
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
        # Show image
        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)


def main():
    rospy.init_node('image_coordinates')
    image_coordinates = ImageCoordinates()
    rospy.spin()


if __name__ == '__main__':
    main()