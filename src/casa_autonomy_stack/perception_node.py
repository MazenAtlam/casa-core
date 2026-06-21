import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('casa_perception_node')
        print("[VISION] Booting Neural Optics & OpenCV Bridge...")
        
        self.subscription = self.create_subscription(
            Image,
            '/ambf/env/cameras/cameraL/ImageData',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        print("[VISION] Waiting for live video feed from AMBF...")

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Convert to HSV Color Space
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 3. Needle Thresholding (Looking for dark/black objects)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([180, 255, 75]) 
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # 4. Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                # THE FIX: Ignore the giant background shadow and tiny noise!
                # Only keep contours that are between 15 and 2000 pixels in size.
                if 15 < area < 2000:
                    valid_contours.append(c)
            
            if valid_contours:
                # Now, the largest object *within the valid size range* is our needle
                needle_contour = max(valid_contours, key=cv2.contourArea)
                
                # Calculate the 2D Centroid of the Needle
                M = cv2.moments(needle_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # 5. Draw the Targeting Reticle
                    cv2.drawContours(cv_image, [needle_contour], -1, (0, 165, 255), 2)
                    cv2.circle(cv_image, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f"NEEDLE LOCKED: ({cX}, {cY})", (cX - 70, cY - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 6. Display the Live Feed
            cv2.imshow("CASA Perception Pipeline (Live Feed)", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Vision Processing Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()