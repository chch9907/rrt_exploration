#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy.linalg import norm
import yaml
from math import pi
import rospy
import time
from functools import partial
from datetime import datetime
import os
# import dynamic_reconfigure.clients
import actionlib
from tf.transformations import quaternion_from_euler
from tf import TransformListener
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Twist#, PointArray
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from copy import copy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from nav.ros_interface import ROS
from rrt_exploration.msg import PointArray  # type: ignore
from distance_map_msgs.msg import DistanceMap  # type: ignore

import tf
from queue import Queue
from nav_utils import (
    index_of_point, bresenham, _world_to_map, _map_to_world, 
    _normalize_heading, WAYPOINT_YAW, OBSTACLE, UNKNOWN,
    get_list_dist, NonBlockingConsole
    )
from tf.transformations import euler_from_quaternion
from global_planner import ompl_plan
from key_params import _params


class MyPlanner:
    def __init__(self, yaw_refine=True):
        rospy.init_node('local_planner', anonymous=False)
        ## topics
        self.goal_topic = rospy.get_param('~goal_topic','/nbv_points')
        self.viewpoint_goal_topic = rospy.get_param('~viewpoint_goal_topic','/viewpoints_goal')
        self.viewpoint_finished_topic = rospy.get_param('~viewpoint_finished_topic','/finished_viewpoint')
        self.planner_topic =rospy.get_param('~planner_topic','/way_point')
        self.speed_topic = rospy.get_param('~speed_topic','/speed')
        self.finished_topic = rospy.get_param('~finished_topic','/finished_goal')
        map_topic = rospy.get_param('~map_topic','/projected_map')
        dist_map_topic = rospy.get_param('~dist_map_topic','/sdf_map_node/distance_field_obstacles')
        query_map_topic = rospy.get_param('~query_map_topic', '/query_map')
        
        ## params and variables
        self.goalpoints = Queue()
        self.viewpoints = Queue()
        self.mapData = OccupancyGrid()
        self.map_array = np.array([])
        self.sdf_map = np.array([])
        self.map_buffer = []
        self.sdf_buffer = []
        self.map_stamp = 0.0
        self.sdf_stamp = 0.0
        self.speed = _params['speed'] #rospy.get_param('~speed', 0.5) # default 2.0
        self.map_frame = rospy.get_param('~map_frame','map')
        self.base_frame = rospy.get_param('~base_frame','base_link')
        self.goal_frame = rospy.get_param('~goal_frame','map')
        self.xyRadius = _params['xyRadius']  #rospy.get_param('~xyRadius', 1.5)
        self.viewpoint_tolerance = _params['viewpoint_tolerance']
        self.zBound = rospy.get_param('~zBound', 5.0)
        rateHz = rospy.get_param('~rateHz', 10)
        self.rate = rospy.Rate(rateHz)
        self.finished_goal_list = []
        self.yaw_refine = yaw_refine
        self.tf_listener = tf.TransformListener()
        self.win_size = rospy.get_param('~win_size', 12)
        self.pose_window = [[0, 0] for _ in range(self.win_size)]
        self.decimal = 2
        self.check_interval = 10  # second
        self.watch_time = 4.0
        self.replan_freq = _params['replan_freq'] # rospy.get_param('~replan_freq', 6.0)
        self.max_replan_num = _params['max_replan_num'] # rospy.get_param('~max_replan_num', 4.0)
        self.path_interval = _params['path_interval']  # rospy.get_param('~path_interval', 2.0)
        validDist = _params['validDist']  # rospy.get_param('~validDist', 3.0)  # 1.5
        self.validDist = validDist
        runTime = rospy.get_param('~runTime', 1)
        plannerType = rospy.get_param('~plannerType', 'informedrrtstar')
        objectiveType = rospy.get_param('~objectiveType', 'weightedlengthandclearancecombo') # PathLength) weightedlengthandclearancecombo
        stepInterval = rospy.get_param('~stepInterval', 2)
        self.global_planner = partial(ompl_plan, 
                                      validDist=validDist, runTime=runTime, plannerType=plannerType, 
                                      objectiveType=objectiveType, stepInterval=stepInterval)
        self.black_areas = _params['black_areas']
        ## subscribers and publishers
        rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
        while not len(self.map_array):
            pass
        rospy.Subscriber(dist_map_topic, DistanceMap, self.dist_map_callback)
        rospy.Subscriber(self.goal_topic, PointArray, self.goal_callback)
        rospy.Subscriber(self.viewpoint_goal_topic, PointArray, self.viewpoint_callback)
        self.sdf_pub = rospy.Publisher(query_map_topic, OccupancyGrid, queue_size=10)
        self.local_planner = rospy.Publisher(self.planner_topic, PointStamped, queue_size=10)
        self.speed_pub = rospy.Publisher(self.speed_topic, Float32, queue_size=10)
        self.finished_pub = rospy.Publisher(self.finished_topic, PointStamped, queue_size=10)
        self.viewpoint_finished_pub = rospy.Publisher(self.viewpoint_finished_topic, PointStamped, queue_size=10)
        
        if self.yaw_refine:
            yaw_topic = rospy.get_param('~yaw_topic', '/my_yaw_cmd_vel')
            self.yaw_pub = rospy.Publisher(yaw_topic, Twist, queue_size=10)
            self.angular_speed = _params['angular_speed']
            self.yaw_tolerance = _params['yaw_tolerance'] # ospy.get_param('~yaw_tolerance', 0.3)
            
        trial_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        self.output_path = f'/home/oem/rrt_explore/src/multi-robot-rrt-exploration-noetic/rrt_exploration/scripts/exp_figs/{trial_id}'
        os.makedirs(self.output_path, exist_ok=True)
        self.nbc = NonBlockingConsole()

    
    def map_callback(self, msg):
        self.mapData = msg
        self.map_array = np.asarray(msg.data).reshape((msg.info.height,
                                        msg.info.width))
        self.map_stamp = msg.header.stamp.to_sec()
        # self.map_buffer.append(copy(map_array))
    
    def get_sdf_map(self, mapData, cur_map_stamp):
        # global_plan_map = OccupancyGrid()
        # global_plan_map.info = self.mapData.info
        # global_plan_map.header = self.mapData.header
        # tuple(self.map_array.ravel())
        self.sdf_pub.publish(mapData)
        st = time.time()
        while self.sdf_stamp != cur_map_stamp and not rospy.is_shutdown():
            # rospy.loginfo(f'wait for sdf map:{cur_map_stamp}, {self.sdf_stamp}')
            self.rate.sleep()
        # rospy.loginfo(f'get sdf map, time:{time.time() - st}')  # 0.01s
        return self.sdf_map
    
    def dist_map_callback(self, msg):
        # self.mapData = msg
        self.sdf_map = np.asarray(msg.data).reshape((msg.info.height,
                                        msg.info.width))
        self.sdf_stamp = msg.header.stamp.to_sec()
        # self.map_buffer.append(copy(self.sdf_map))
        
        # self.sdf_map[np.where(self.sdf_map > 0)] = 0  # negative values lie in unknown spaces
        # plt.matshow(self.sdf_map)
        # plt.colorbar()
        # plt.savefig('/home/oem/rrt_explore/src/multi-robot-rrt-exploration-noetic/rrt_exploration/scripts/dist_map.png')
        # plt.cla()
        if self.sdf_map.shape != self.map_array.shape:
            rospy.logwarn(f'dist:{self.sdf_map.shape}, {self.sdf_stamp} | map:{self.map_array.shape}, {self.map_stamp}')
        
    def get_tf(self, duration=5.0):
        try:
            self.tf_listener.waitForTransform(self.map_frame,
                                            self.base_frame,
                                            rospy.Time(0),
                                            rospy.Duration(duration))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame,
                                                            self.base_frame,
                                                            rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo(f"TF exception, cannot get odom pose. {e}")
            return
        point = Point(*trans)
        # rospy.loginfo([point.x, point.y, _normalize_heading(_quat_to_angle(*rot))])
        _, _, yaw = euler_from_quaternion(rot)
        return np.array([point.x, point.y, yaw])

    def compensate_yaw(self, goal_dyaw, cur_yaw):
        # rospy.loginfo('compensate_yaw')
        # rospy.loginfo(f'compensate dyaw:{goal_dyaw}')
        pre_angle = cur_yaw
        acc_angle_turn = 0
        t = 0
        yaw_msg = Twist()
        yaw_msg.angular.z = self.angular_speed if goal_dyaw > 0 else -self.angular_speed
        while abs(goal_dyaw) - abs(acc_angle_turn) > self.yaw_tolerance \
            and not rospy.is_shutdown() and self.nbc.get_data() != '\n':
            self.yaw_pub.publish(yaw_msg)
            self.rate.sleep()
            cur_yaw = self.get_tf()[2]
            delta_angle = _normalize_heading(cur_yaw - pre_angle)
            acc_angle_turn += abs(delta_angle)
            pre_angle = cur_yaw
            t += 1
            # rospy.loginfo('acc_angle_turn:{}, goal_dyaw:{}'.format(
            #     acc_angle_turn, abs(goal_dyaw)))

        stop_msg = Twist()
        stop_msg.angular.z = 0
        self.yaw_pub.publish(stop_msg)
        rospy.loginfo(f'finish dyaw compensatation,cur_yaw:{cur_yaw}')  #!bug

    def viewpoint_callback(self, msg):
        # with self.goalpoints.mutex:
        #     self.goalpoints.queue.clear()
        for point in msg.points:
            rospy.loginfo(f'local planner get viewpoint:{[point.x, point.y, point.z]}')
            self.viewpoints.put([point.x, point.y, point.z])  # x, y, yaw
    
    def goal_callback(self, msg):
        with self.goalpoints.mutex:
            self.goalpoints.queue.clear()
        for point in msg.points:
            # rospy.loginfo(f'local_planner get goal:{[point.x, point.y, point.z]}')
            self.goalpoints.put([point.x, point.y, point.z])  # x, y, yaw
    
    
    def goal_wrapper(self, goal, finished=False):
        goal_msg = PointStamped()
        goal_msg.header.frame_id = self.goal_frame
        goal_msg.header.stamp = rospy.Time().now()
        goal_msg.point.x = goal[0] 
        goal_msg.point.y = goal[1]
        if finished:
            goal_msg.point.z = goal[2]  #!bug: z can only be 0
        return goal_msg

    def reachable(self, cur_pose, min_move=1):
        return np.max(norm(cur_pose - \
                           np.array(self.pose_window), axis=1)) > min_move           

    def check_finish(self, cur_pose, next_waypoint, next_goal, isFrontier):
        next_waypoint_map = _world_to_map(self.mapData, next_waypoint)
        value = self.map_array[next_waypoint_map]
        if value == OBSTACLE:
        # cur_pose_map = _world_to_map(self.mapData, cur_pose)
        # if not self.in_line_of_sight(cur_pose_map, next_waypoint_map):
            rospy.loginfo(f'waypoint {np.round(next_waypoint, self.decimal)} is obstacle')
            # self.finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
            # rospy.loginfo(f'pub finished point:{next_goal}')
            return True, True

        dist_to_waypoint = norm(cur_pose[:2] - np.array(next_waypoint[:2]))
        # rospy.loginfo(f'dist:{norm(cur_pose[:2] - np.array(next_waypoint[:2]))}')
        
        if isFrontier:
            next_goal_map = _world_to_map(self.mapData, next_goal)
            frontier_value = self.map_array[next_goal_map]
            dist_to_goal = norm(cur_pose[:2] - np.array(next_goal[:2]))
            # if dist_to_goal <= 2.5 * self.xyRadius:
            #     self.finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
            #     rospy.loginfo(f'pub finished point:{next_goal}')
            return dist_to_waypoint <= self.xyRadius, dist_to_goal <= 2. * self.xyRadius #  frontier_value != UNKNOWN
        else:
            # dist_constrain = norm(cur_pose[:2] - np.array(next_waypoint[:2])) < self.xyRadius 
                    # and cur_pose[2] < self.zBound
            # rospy.loginfo(f'dist:{dist_constrain}')
            return dist_to_waypoint <= self.viewpoint_tolerance, False

    def in_line_of_sight(self, cur_pose_map, goal_map):
        lines = list(bresenham(cur_pose_map, goal_map))[1:-1]  # remove current and goal
        # rospy.loginfo(lines)
        # lines = [slice(line) for line in lines]
        values = [self.map_array[p] for p in lines]
        # values = self.map_array[np.array(lines)]
        # rospy.loginfo(f'values:{values}:{(np.array(values) == 0).all()}')
        return (np.array(values) == 0).all()

    
    def plot_planned_path(self, plan_path, cur_pose_map, next_goal, next_goal_map, 
                          goal_idx, kth_replan, title='', size = 1):
        plt.plot(plan_path[:, 1], plan_path[:, 0], color='purple')
        plt.scatter(plan_path[:, 1], plan_path[:, 0], color='green', marker='*')
        plt.scatter(cur_pose_map[1], cur_pose_map[0], color='blue', marker='*')
        obstacles = np.where(self.map_array == OBSTACLE)
        unknowns = np.where(self.map_array == UNKNOWN)
        
        height = self.map_array.shape[0]
        plt.text(next_goal_map[1], next_goal_map[0], f'{next_goal}')
        plt.scatter(obstacles[1], obstacles[0], color='black', s=size, label='obstacles')
        plt.scatter(unknowns[1], unknowns[0], color='gray', s=size, label='unknowns')
        plt.title(title)
        plt.savefig(f'{self.output_path}/path_{goal_idx}{kth_replan}.png')
        plt.cla()
    
    def path_planning(self, cur_pose, next_goal, goal_idx, kth_replan='', show=True, isViewpoint=False):
        cur_pose_map = _world_to_map(self.mapData, cur_pose)
        next_goal_map = _world_to_map(self.mapData, next_goal)
        # rospy.loginfo(f'next_goal_map:{next_goal_map}')
        # route_to_next_map = Queue()
        route_to_next = Queue()
        if self.in_line_of_sight(cur_pose_map, next_goal_map):
            rospy.loginfo(f'in line of sight')
            if not len(kth_replan):
                route_to_next.put(next_goal) 
                # route_to_next_map.put(next_goal_map)
            if show:
                plan_path = np.array([cur_pose_map, next_goal_map])
                self.plot_planned_path(plan_path, cur_pose_map, next_goal, 
                                       next_goal_map, goal_idx, kth_replan, title=f'in line of sight, isViewpoint:{isViewpoint}')
        else:
            sdf_map = self.get_sdf_map(self.mapData, self.map_stamp)
            ## plot sdf map
            # fig, ax = plt.subplots()
            # ax.matshow(sdf_map)
            # plt.savefig(f'{self.output_path}/sdf_{goal_idx}{kth_replan}.png')
            # ax = plt.gca()
            # ax.clear()
            raw_plan_path = self.global_planner(cur_pose_map, next_goal_map, sdf_map, 
                                                self.mapData, self.black_areas)
            if raw_plan_path is None:
                rospy.loginfo('no solution path, abort')
                if show:
                    plan_path = np.array([cur_pose_map, next_goal_map])
                    self.plot_planned_path(plan_path, cur_pose_map, next_goal, 
                                       next_goal_map, goal_idx, kth_replan, title=f'no solution path, abort, isViewpoint:{isViewpoint}')
            else:
                # rospy.loginfo(f'raw solution path len:{len(raw_plan_path)}')
                start = False
                pre_point = cur_pose[:2]
                inter_dist = []
                plan_path = []
                plan_path_world = []
                # raw_plan_path_world = []
                # rospy.loginfo(f'raw_plan:{raw_plan_path[-1]}, next_goal_map:{next_goal_map}')
                for point in raw_plan_path:
                    point_world = _map_to_world(self.mapData, point)
                    # raw_plan_path_world.append(point_world)
                    # if not start and norm(np.array(point_world) - cur_pose[:2]) < interval:
                    #     continue
                    # else:
                    dist = get_list_dist(point_world, pre_point)
                    if dist < self.path_interval:
                        continue
                    else:
                        plan_path_world.append(point_world)
                        plan_path.append(point)
                        # plan_path_world.append(point_world)
                        # if pre_point is not None:
                        inter_dist.append(dist)
                        pre_point = point_world
                        # pre_point = point_world
                        route_to_next.put(point_world + [WAYPOINT_YAW])
                        # route_to_next_map.put(point)
                # if not len(plan_path) or get_list_dist(plan_path_world[-1], next_goal) > self.xyRadius:
                dist_to_goal = get_list_dist(pre_point, next_goal)
                if next_goal_map not in plan_path and dist_to_goal > self.xyRadius:
                    plan_path.append(next_goal_map)
                    route_to_next.put(next_goal)
                    # route_to_next_map.put(next_goal_map)
                    inter_dist.append(dist_to_goal)
                
                ## check sdf on the route
                # Nmax = 10 
                height = sdf_map.shape[0]
                for j in range(len(raw_plan_path) - 1):
                    lines = list(bresenham(raw_plan_path[j], raw_plan_path[j + 1]))
                    values = [sdf_map[height - p[0] - 1, p[1]] for p in lines]
                    wrong_values = [sdf for sdf in values if sdf < self.validDist]
                    if len(wrong_values):
                        rospy.loginfo(f'wrong values:{wrong_values}')

                # rospy.loginfo(f'discretized plan_path:{plan_path}')
                # rospy.loginfo(f'inter dist among planned path:{inter_dist}')
                if show and len(plan_path):
                    plan_path = np.array([cur_pose_map] + plan_path + [next_goal_map])
                    self.plot_planned_path(plan_path, cur_pose_map, next_goal, 
                                       next_goal_map, goal_idx, kth_replan, title=f'solution path, isViewpoint:{isViewpoint}')
                    # raw_plan_path = np.array(raw_plan_path)
                    # rospy.loginfo(f'plan_path shape:{plan_path.shape}')
                    # plt.plot(raw_plan_path[:, 1], raw_plan_path[:, 0], color='purple')
                    # plt.scatter(raw_plan_path[:, 1], raw_plan_path[:, 0], color='green', marker='*')

        return route_to_next  #, route_to_next_map
    
    def run(self, ):
        isWaiting = False
        next_waypoint = []
        next_waypoint_map = []
        next_goal = []
        # route_to_next_map = Queue()
        # route_to_next = Queue()
        # step = 0
        isFrontier = False
        isViewpoint = False
        isReplan = False
        goal_idx = 0
        pre_pose = self.get_tf()
        pre_stamp = self.map_stamp
        start_time = time.time()
        kth_replan = 1
        pre_isViewpoint = False
        while not rospy.is_shutdown():
            if not isWaiting and self.goalpoints.empty() and self.viewpoints.empty(): # and self.viewpoints.empty():
                # rospy.loginfo('currently no candidate')
                # rospy.loginfo(f'{isWaiting}, {self.goalpoints.empty()}')
                # self.rate.sleep()
                continue
            cur_pose = self.get_tf()
            cur_stamp = self.map_stamp
            if isWaiting:
                ########################### check if finished and get next waypoint ###########################
                #! for online 
                isfinished, isknown_or_obstacle = self.check_finish(cur_pose, next_waypoint, 
                                                    next_goal, isFrontier)
                
                #! only for offline testing
                if self.nbc.get_data() == '\n':# or route_to_next.empty():
                    rospy.loginfo('manual abort')
                    isfinished = True
                    isknown_or_obstacle = True
                # else:
                #     isfinished = False
                #     isknown_or_obstacle = False
                
                if not isfinished and kth_replan <= ((time.time() - start_time) // self.replan_freq) < self.max_replan_num:
                    rospy.loginfo(f'--goal {goal_idx}: {kth_replan}th replanning--')
                    route_to_next = self.path_planning(cur_pose, next_goal, goal_idx, f'-{kth_replan}', isViewpoint=isViewpoint)
                    rospy.loginfo('finish replanning')
                    kth_replan += 1
                    isfinished = True
                    # isReplan = True
                
                if isfinished:
                    rospy.loginfo(f'reach {goal_idx}th goal: {np.round(next_waypoint, self.decimal)}')
                    isAbort = False
                    if (isknown_or_obstacle and not route_to_next.empty()) or kth_replan == self.max_replan_num:
                        route_to_next = Queue()
                        # route_to_next_map = Queue()
                        isAbort = True                 
                    
                    # if not self.viewpoints.empty() and isViewpoint == False:
                    #     next_waypoint = self.viewpoints.get()
                    #     next_waypoint_map = _world_to_map(self.mapData, next_waypoint)
                    #     # rospy.loginfo(f'pub waypoint:{next_waypoint}')
                    #     self.local_planner.publish(self.goal_wrapper(next_waypoint))
                    #     self.speed_pub.publish(self.speed)
                    #     isViewpoint = True
                    #     rospy.loginfo(f'next goal: viewpoint {next_goal}!!!!!!')  
                    if not route_to_next.empty():
                        next_waypoint = route_to_next.get()
                        # next_waypoint_map = route_to_next_map.get()
                        rospy.loginfo(f'pub waypoint:{next_waypoint}, cur_pose:{cur_pose}')
                        self.local_planner.publish(self.goal_wrapper(next_waypoint))
                        self.speed_pub.publish(self.speed)
                        start_time = time.time()
                    else:  ## finished 
                        self.finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
                        rospy.loginfo(f'pub finished point:{next_goal}')
                        if isViewpoint:
                            stop_msg = self.goal_wrapper(cur_pose)
                            self.local_planner.publish(stop_msg)
                            goal_dyaw = _normalize_heading(next_goal[2] - cur_pose[2])
                            rospy.loginfo(f'viewpoint compsensate_yaw:{goal_dyaw}, {cur_pose[2]}')
                            self.compensate_yaw(goal_dyaw, cur_pose[2])
                            
                            self.viewpoint_finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
                            rospy.sleep(3)
                        # elif self.yaw_refine:
                        #     if next_waypoint[2] == WAYPOINT_YAW or isAbort:
                        #         # goal_dyaw = 0
                        #         pass
                        #     else:
                        #         stop_msg = self.goal_wrapper(cur_pose)
                        #         self.local_planner.publish(stop_msg)
                        #         # self.rate.sleep()
                        #         goal_dyaw = _normalize_heading(next_goal[2] - cur_pose[2])
                        #         rospy.loginfo(f'frontier compsensate_yaw:{goal_dyaw}')
                        #         # rospy.loginfo(f'pub dyaw:{dyaw}, {next_waypoint[2]}, {cur_pose[2]}')
                        #         self.compensate_yaw(goal_dyaw, cur_pose[2])
                        #         rospy.sleep(0.5)

                        self.finished_goal_list.append(copy(next_goal))
                        rospy.loginfo(f'finish {len(self.finished_goal_list)} waypoint, isFrontier:{isFrontier}, isViewpoint:{isViewpoint}')
                        isWaiting = False
            else:
                ########################### get next subgoal ###########################
                goal_idx += 1
                rospy.loginfo(f'------goal_idx:{goal_idx}------')
                next_goal = []
                rospy.loginfo('get next goal')
                if not self.viewpoints.empty():# and pre_isViewpoint is False:
                    while not self.viewpoints.empty():
                        next_goal = self.viewpoints.get()
                        rospy.loginfo('next viewpoint has already finished')
                        if next_goal not in self.finished_goal_list:
                            break
                    if not len(next_goal):
                        rospy.loginfo(f'currently no viewpoints')
                        continue
                    else:
                        isViewpoint = True
                        rospy.loginfo(f'next goal: viewpoint {next_goal}!!!!!!')
                else:
                    while not self.goalpoints.empty():
                        next_goal = self.goalpoints.get()
                        rospy.loginfo('next frontier has already finished')
                        if next_goal not in self.finished_goal_list:
                            break
                    if not len(next_goal):
                        rospy.loginfo(f'currently no goal points')
                        continue
                    if next_goal[2] == WAYPOINT_YAW:
                        isFrontier = False
                        rospy.loginfo(f'next goal: intermediate {next_goal}')
                    else:
                        isFrontier = True
                        rospy.loginfo(f'next goal: frontier {next_goal}')
                    isViewpoint = False
                
                # rospy.loginfo(f'pub finished point:{next_goal}')
                # self.finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
                route_to_next = self.path_planning(cur_pose, next_goal, goal_idx, isViewpoint=isViewpoint)

                               
                # if not route_to_next.empty():
                #     continue
                    # rospy.loginfo('no route, send cur_pose as goal')
                    # next_waypoint = cur_pose
                    # next_waypoint[2] = WAYPOINT_YAW * 2
                    # next_waypoint_map = _world_to_map(self.mapData, cur_pose)
                    # else:
                if route_to_next.empty():
                    self.finished_pub.publish(self.goal_wrapper(next_goal, finished=True))
                    rospy.loginfo(f'pub finished point:{next_goal}')
                    continue
                next_waypoint = route_to_next.get()
                # next_waypoint_map = route_to_next_map.get()
                rospy.loginfo(f'pub waypoint:{next_waypoint}, cur_pose:{cur_pose}')
                self.local_planner.publish(self.goal_wrapper(next_waypoint))
                self.speed_pub.publish(self.speed)
                isWaiting = True
                # step += 1 
                
                start_time = time.time()
                kth_replan = 1
                pre_isViewpoint = isViewpoint
                
            # self.rate.sleep()
            

if __name__ == '__main__':
    # yaw_refine = rospy.get_param('~yaw_refine', yaw_refine)
    node = MyPlanner(yaw_refine=True)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass