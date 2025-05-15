#!/usr/bin python3

# Copyright (c) 2024 System Dynamics Lab (SDL) at Pusan National University
# , South Korea (PNU)
# SDL Homepage, see <https://sites.google.com/view/sysdyn/home?authuser=0>
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 
# Professor : Seunghun Baek
# Authors   : Jaewon Lee  (M.S student at SDL), email: risiowon@pusan.ac.kr
#             Jaewoon Lee (M.S student at SDL), email: jaewoon99@pusan.ac.kr

#%% modules
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
import numpy as np
import argparse

#%% class
class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller_node')
        # ============ Subscriber ============
        self.odom_sub = self.create_subscription(
                                 Odometry,
                                 '/pf/pose/odom',
                                 self.odom_callback,
                                 10)

        # =========== Publisher ============
        self.control_cmd_pub = self.create_publisher(
                                    AckermannDriveStamped,
                                    "/drive",
                                    10)
        
        self.global_path_pub = self.create_publisher(Path, 'global_path', 10)
        
        # ========= Timer ===========
        timer_period = 0.1  # seconds = 20Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # ============ variables ==============
        self.current_x   = None # current x position of the vehicle
        self.current_y   = None # current y position of the vehicle
        self.current_yaw = None # current yaw orientation of the vehicle
        self.current_v   = None # current velocity of the vehicle
        self.prev_t      = None
        self.error_v     = None # velocity error for longitudinal speed control
        
        # <<<  
        #TODO: tuning gain parameters
        # Stanley params
        self.lookahead_wp = 2  # the number of lookahead waypoints
        self.L = 1.2    # vehicle CoG to front axle
        self.k = 3.0            # stanley gain parameter
        self.ks = 0.0
        self.stanley_overall_gain = 0.5
        # >>>

        # [path]
        self.global_path = None

        # Result params
        self.total_cte = 0.0    # total cross-track error
        self.elapsed_time = 0.0 # elapsed time until goal

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v  = np.sqrt(vx**2 + vy**2) # m/s

        self.current_v = v
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Quaternion to Euler conversion
        orientation_list = [msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)        

    def set_steer_cmd(self, input_steer_in_rad):
                 
        input_steer = np.rad2deg(input_steer_in_rad) # (1) write here   
        input_steer = np.fmin(np.fmax(input_steer, -70), 70) # (2) write here
        output_steer = input_steer / 70 # (3) write here
        return output_steer

    def timer_callback(self):
        '''
        Validating for Stanley controller
        '''
        current_x = self.current_x
        current_y = self.current_y
        current_yaw = self.current_yaw
        current_v = self.current_v
        current_t = self.get_clock().now().nanoseconds * 1e-9

        if current_x is not None and current_y is not None and current_yaw is not None and current_v is not None and self.global_path is not None:
            
            # Stanley controller
            dt = current_t - self.prev_t if self.prev_t is not None else 0.1
            current_frontaxle_x = current_x + self.L * np.cos(current_yaw)
            current_frontaxle_y = current_y + self.L * np.sin(current_yaw)

            terminal_idx = len(self.global_path)-1
            closest_wp_idx = self.get_closest_waypoint(self.global_path, current_frontaxle_x, current_frontaxle_y)
            
            if closest_wp_idx > terminal_idx or closest_wp_idx == terminal_idx:
                # <<<
                #TODO: publish control msg 
                control_msg = AckermannDriveStamped()
                control_msg.header.stamp = self.get_clock().now().to_msg()
                #control_msg.header.frame_id = "base_link"
                control_msg.drive.speed = 1.0                # Set the desired velocity value.
                control_msg.drive.steering_angle = 0   
                control_msg.drive.steering_angle_velocity = 0.5

                
                self.control_cmd_pub.publish(control_msg)
                # ==================================== >>>

                self.get_logger().info('************************************************************')
                self.get_logger().info('Global path tracking is done by stanley lateral control')
                self.get_logger().info(f'Total CTE: {self.total_cte:.2f}, Elapsed_time: {self.elapsed_time:.2f}')
                self.get_logger().info(f"[Control Info]")
                self.get_logger().info(f'Speed: "{control_msg.drive.speed:.2f}", Steer: "{control_msg.drive.steering_angle:.2f}"')
                self.get_logger().info(f"Hz: {1/dt:.2f}")


                # Shutdown the ROS 2 node
                rclpy.shutdown()

            else:
                # <<<
                #TODO: Stanley control theory
                closest_x = self.global_path[closest_wp_idx][0] # (1) write here
                closest_y = self.global_path[closest_wp_idx][1] # (1) write here    
                dx = self.global_path[closest_wp_idx + 1][0] -  closest_x # (2) write here
                dy = self.global_path[closest_wp_idx + 1][1] -  closest_y # (2) write here
                closest_yaw = np.arctan2(dy, dx) # (3) write here [-pi,pi]

                """
		        Description:
		        - Calculate the cross product of the vector from the current position to the closest waypoint and the vector from the closest waypoint to the next waypoint.
		        - If the cross product is positive, the path is on the left side of the vehicle.
		        - If the cross product is negative, the path is on the right side of the vehicle.

		        Hints:
		        - (4) Calculate the vector from the current position to the closest waypoint.
		               If you want to calculate the vector, you can use 'np.array'. Ex) vector = np.array([x2-x1, y2-y1])
		               Frontal axle's coordinates: (current_frontaxle_x, current_frontaxle_y), check line 231, 232.
		        - (5) Calculate the vector from the closest waypoint to the next waypoint.
		        - (6) Calculate the direction of the path.
		              For cross product, use 'np.cross()'.
		              If the cross product is positive(+), then direction is 1. (use 'np.sign()')
		              If the cross product is negative(-), then direction is -1. (use 'np.sign()')
		        """
                cte_vec = np.array([closest_x - current_frontaxle_x, closest_y - current_frontaxle_y]) # (4) write here
                closest_vec = np.array([dx, dy]) # (5) write here
                direction = np.sign(np.cross(cte_vec, closest_vec)) # (6) write here

                """
		        Description:
		        - Calculate the cross-track error(cte).
		        - Calculate the yaw error [-pi,pi].
		        - Calculate the steering angle [-1,1].

		        Hints:
		        - (7) Calculate the cross-track error by using 'np.linalg.norm()' which can calculate the length of the vector.
		        - (8) Use 'self.normalize_angle' function you've implemented in [TODO.2].
		        - (9) Calculate the desired steering angle by using 'np.arctan2()' for tan^(-1).
		              *** Use the formula of Stanley controller.
		        - (10) Normalize the steering angle [-1,1] by using 'self.set_steer_cmd' function you've implemented in [TODO.3].
		        """
                cte = np.linalg.norm(cte_vec) # (7) write here
                yaw_error = self.normalize_angle(current_yaw - closest_yaw) # (8) write here
                steer = self.stanley_overall_gain*(yaw_error - np.arctan2(self.k*cte, current_v + self.ks)*direction)                
                steer_output = self.set_steer_cmd(steer)
		        # >>>	
             
                # <<<
                # TODO: publish control msg
                control_msg = AckermannDriveStamped()
                control_msg.header.stamp = self.get_clock().now().to_msg()
                control_msg.drive.speed = 0.5 # throttle : [0., 1.]
                #control_msg.brake    = 0.0    # brake    : [0., 1.]
                control_msg.drive.steering_angle = steer_output    # CARLA steering : [-1, 1], (+) : clock-wise, (-) : anti clockwise
                control_msg.drive.steering_angle_velocity = steer_output
                self.control_cmd_pub.publish(control_msg)
                # >>>

                # logger
                self.get_logger().info('************************************************************')
                self.get_logger().info(f"[Lateral Info]")
                self.get_logger().info(f"Cross-track error: {cte:.2f}m, Yaw error: {yaw_error:.2f}, Direction: {direction}")
                self.get_logger().info(f'Total CTE: {self.total_cte:.2f}m')
                self.get_logger().info(f"[Control Info]")
                self.get_logger().info(f'Speed: "{control_msg.drive.speed:.2f}", Steer: "{control_msg.drive.steering_angle:.2f}"')
                self.get_logger().info(f"Hz: {1/dt:.2f}")

                # save params
                self.total_cte += abs(cte)
                self.elapsed_time += dt

            self.prev_t = current_t
        
        else:
            self.get_logger().info('************************************************************')
            self.get_logger().info('There is no execution.')

    def get_closest_waypoint(self, global_path, current_x, current_y):
        min_len = 1e9
        closest_wp_idx = 0

        for i in range(len(global_path)):
            dist = np.sqrt((current_x - global_path[i,0])**2 + (current_y - global_path[i,1])**2)

            if dist < min_len:
                min_len = dist
                closest_wp_idx = i

        return closest_wp_idx
    
    def normalize_angle(self, angle):
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi

        return angle

    def publish_global_path(self):
        if self.global_path is not None:
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'

            poses = []
            for point in self.global_path:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.position.z = 0.0 

                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0

                poses.append(pose)
            
            path_msg.poses = poses
            self.global_path_pub.publish(path_msg)
            self.get_logger().info('Published global path.')
        else:
            self.get_logger().warn('Global path is not loaded.')
    
#%% main
def main(args=None):
    rclpy.init(args=None)
    node = StanleyController()

    parser = argparse.ArgumentParser(description='Load and plot a smooth global path from a .npy file.')
    parser.add_argument('npy_file', type=str, help='Path to the smooth_global_path.npy file')
    args = parser.parse_args()
    node.global_path = np.load(args.npy_file)

    node.publish_global_path()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
