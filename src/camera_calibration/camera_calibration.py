#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageCoordinates:

    def __init__(self):
        rospy.loginfo("Running image coordinates node...")
        self.cv_image = None
        self.resolution = (960, 528)
        self.n_points = 0
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
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
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
            self.coordinates.append([[(x - self.resolution[0]/2)/self.resolution[0], 
                                      (y - self.resolution[1]/2)/self.resolution[1]], 
                                      [float(x_w), float(y_w), float(z_w)]])
            self.n_points += 1
            rospy.loginfo(f"World coordinates: ({x_w}, {y_w}, {z_w})")
            rospy.loginfo("-"*50)
            cont = input("Continue? (y/n): ")
            if cont == "n":
                rospy.loginfo("-"*50)
                rospy.loginfo("Coordinates saved")
                rospy.loginfo(self.coordinates)
                rospy.loginfo("-"*50)
                self.run_DLT()

    def run_DLT(self):
        """
        Run Direct Linear Transformation (DLT) algorithm
        """
        rospy.loginfo("Running DLT algorithm")
        A = np.zeros((self.n_points*2, 12))
        for i in range(self.n_points):
            x, y = self.coordinates[i][0]
            X, Y, Z = self.coordinates[i][1]
            A[i*2, 0:4] = [-X, -Y, -Z, -1]
            A[i*2, 8:12] = [x*X, x*Y, x*Z, x]
            A[i*2+1, 4:8] = [-X, -Y, -Z, -1]
            A[i*2+1, 8:12] = [y*X, y*Y, y*Z, y]
        # Solve for the projection matrix
        _, _, V = np.linalg.svd(A)
        solution = V[-1, :]
        print(solution)
        solution = solution/solution[-1]
        print(solution)
        P = np.reshape(solution, (3, 4))
        rospy.loginfo("Projection matrix:")
        rospy.loginfo(P)
        rospy.loginfo("-"*50)
        self.save_projection_matrix(P)

    def save_projection_matrix(self, P):
        """
        Save the projection matrix to a file
        """
        rospy.loginfo("Saving projection matrix to file")
        with open(rospkg.RosPack().get_path("prediction") + "/data/projection_matrix.csv", "w") as file:
            for row in P:
                file.write(",".join([str(x) for x in row]) + "\n")
        rospy.loginfo("Projection matrix saved to file")
        rospy.loginfo("-"*50)
        rospy.loginfo("Calibration completed")
        rospy.loginfo("-"*50)
        rospy.signal_shutdown("Calibration completed")



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