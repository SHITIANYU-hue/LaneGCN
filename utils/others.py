#!/usr/bin/python
# -*- coding: UTF-8 -*-
# __all__ = ['process_back', 'process_input']
import numpy as np
from utils.frenet_optimal_trajectory import generate_target_course,frenet_optimal_planning,calc_frenet_paths
from typing import Any, Tuple

def process_input(data):
    data_=data.groupby("TRACK_ID")
    for group_name, group_data in data_:
        object_type = group_data["OBJECT_TYPE"].values[0]
        if object_type == "AGENT":
            cor_x = group_data["X"].values
            cor_y = group_data["Y"].values
    input_data=[]
    for i in range(len(cor_x)):
        input_data.append([cor_x[i],cor_y[i]])
    input_data=np.asarray(input_data)
    city_name=data["CITY_NAME"].values[0]

    return city_name, input_data

def get_mean_velocity(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean velocity of the observed trajectory.

    Args:
        coords: Coordinates for the trajectory
    Returns:
        Mean velocity along x and y

    """
    vx, vy = (
        np.zeros((coords.shape[0], coords.shape[1] - 1)),
        np.zeros((coords.shape[0], coords.shape[1] - 1)),
    )

    for i in range(1, coords.shape[1]):
        vx[:, i - 1] = (coords[:, i, 0] - coords[:, i - 1, 0]) / 0.1
        vy[:, i - 1] = (coords[:, i, 1] - coords[:, i - 1, 1]) / 0.1
    vx = np.mean(vx, axis=1)
    vy = np.mean(vy, axis=1)

    return vx, vy

def process_xy(input):
    x=[]
    y=[]
    for i in range(len(input)):
        x.append(input[i][0])
        y.append(input[i][1])
        
    return np.asarray(x),np.asarray(y)

def cal_speed_list(data):
    x,y=process_xy(data)
    d_x,d_y=np.gradient(x),np.gradient(y)
    velocity=[]
    import math
    for i in range(len(d_x)):
        v=(1/0.1)*math.sqrt(d_x[i]**2+d_y[i]**2)
        velocity.append(v)
    return velocity


def sample_speed(data):
    if data >33:
        data=33
    v_range_l=max(0,data-6*1)
    v_range_r=min(33,data+6*1) ## we use average speed here to calculate
    speed_data=np.arange(v_range_l,v_range_r,0.005)
    
    return np.random.choice(speed_data)

def process_xy_back(x,y):
    data=[]
    for i in range(len(x)):
        data.append([x[i],y[i]])
    return np.asarray(data)

def inc_length(data,up_to=50,ratio=1):
    length_cur=len(data)
    if length_cur>up_to:
        return data[0:up_to]
    else:
        x,y=process_xy(data)
        d_x,d_y=ratio*np.gradient(x),ratio*np.gradient(y)

        for i in range(up_to-length_cur):
            x=np.append(x,2*d_x[-1]+x[-1])
            y=np.append(y,2*d_y[-1]+y[-1])
            
        data=process_xy_back(x,y)
        return data

def get_direction(trajectory,points=2):
    velocity_x_mps = []
    velocity_y_mps = []
    for i in range(1,points,1):
        velocity_x_mps.append(trajectory[-i,0] - trajectory[-(1+i),0])
        velocity_y_mps.append(trajectory[-i,1] - trajectory[-(1+i),1])
    return velocity_x_mps,velocity_y_mps

def prediction(trajectory,num_points=30,avg_points=1):
    #a simple prediction function that predict straight line with constant velocity
    velocity_x_mps = []
    velocity_y_mps = []
    for i in range(1,avg_points+1,1):
        velocity_x_mps.append(trajectory[-i,0] - trajectory[-(1+i),0])
        velocity_y_mps.append(trajectory[-i,1] - trajectory[-(1+i),1])
        
    # velocity_x_mps = np.mean(velocity_x_mps)
    # velocity_y_mps = np.mean(velocity_y_mps)
    
    velocity_x_mps = sample_speed(velocity_x_mps)
    velocity_y_mps = np.mean(velocity_y_mps)
    current_traj = trajectory[-1]
    results = np.zeros((len(trajectory)+num_points,2))
    
    results[0:len(trajectory)] = trajectory
    
    for i in range(num_points):
        results[len(trajectory)+i] = np.array([current_traj[0]+velocity_x_mps,current_traj[1]+velocity_y_mps])
        current_traj = results[len(trajectory)+i]
    return results

def generate_offset(prune_dfs,offset=1,l_r=True):
    
    from shapely.geometry import Point, LineString
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

        xN,yN= process_xy(prune_dfs)
        interp_points=[]
        for i in range(len(xN)):
            interp_points.append(Point(xN[i],yN[i]))
        line=LineString(interp_points)
        if l_r:
            after_paraell=np.asarray(line.parallel_offset(offset,'right'))
        else:
            after_paraell=np.asarray(line.parallel_offset(offset,'left'))
    return after_paraell

def convert_to_frenet(prune_dfs):
    x,y=process_xy(prune_dfs)

    x_=[]
    y_=[]
    for i in range(0,len(x),2): 
        x_.append(x[i])
        y_.append(y[i])


    tx, ty, tyaw, tc, csp = generate_target_course(x_, y_)
    
    return tx, ty, tyaw, tc, csp 

def close_list_index(a,b):
    distance=[]
    for i in range(len(a)):
        dis_x=(a[i][0]-b[0][0])**2
        dis_y=(a[i][1]-b[0][1])**2
        dist=dis_x+dis_y
        distance.append(dist)
        
    
    return distance.index(min(distance))

def cal_dist(a,b):
    import math
    dist = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return dist

def cal_paral(prune_dfs_,prune_dfs_2):
    prune_dfs_new=[]
    for i in range(min(len(prune_dfs_),len(prune_dfs_2))):
        # print(prune_dfs_[i])
        x=2*prune_dfs_[i][0]-prune_dfs_2[i][0]
        y=2*prune_dfs_[i][1]-prune_dfs_2[i][1]
        prune_dfs_new.append([x,y])

    return prune_dfs_new