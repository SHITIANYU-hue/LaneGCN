from logging import root
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from utils.map_features_utils import MapFeaturesUtils
from utils.visualize_sequences import  interpolate_polyline, viz_sequence
import pickle as p
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
from utils.others import process_input,process_xy,process_xy_back
from pykalman import KalmanFilter
from scipy import poly1d
from pykalman import KalmanFilter
from argoverse.map_representation.map_api import ArgoverseMap
from utils.others import process_input, process_xy, sample_speed, process_xy_back, get_direction, prediction,generate_offset,convert_to_frenet,close_list_index,cal_dist,cal_paral,cal_speed_list,inc_length
from utils.frenet_optimal_trajectory import generate_target_course,frenet_optimal_planning,calc_frenet_paths
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from multiprocessing import Pool
from itertools import product



def model_generator(data,start):

    city_name_, data_= process_input(data)

    ## filter the data

    x,y=process_xy(data_)
    measurements=data_
    initial_state_mean = [measurements[0, 0],
                        0,
                        measurements[0, 1],
                        0]

    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]


    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    # plt.figure(1)
    # times = range(measurements.shape[0])
    # plt.plot(times, measurements[:, 0], 'bo',
    #          times, measurements[:, 1], 'ro',
    #          times, smoothed_state_means[:, 0], 'b--',
    #          times, smoothed_state_means[:, 2], 'r--',)
    # plt.show()
    kf2 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    observation_covariance = 10*kf1.observation_covariance,
                    em_vars=['transition_covariance', 'initial_state_covariance'])

    kf2 = kf2.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
    data_k=process_xy_back(smoothed_state_means[:, 0],smoothed_state_means[:, 2])

    ## dfs search
    avm = ArgoverseMap()
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(data_k[0:19], city_name_, viz=True)
    dfs=candidate_centerlines


    ## dfs pruning
    dfs_save_p=[]
    for i in range(len(candidate_centerlines)):
        x,y=process_xy(candidate_centerlines[i])
        # x_=np.asarray(x)-2000
        # y_=0.001*np.asarray(y)-2000
        x_=[]
        y_=[]
        for i in range(0,len(x),2): 
            x_.append(x[i])
            y_.append(y[i])


        tx, ty, tyaw, tc, csp = generate_target_course(x_, y_)
        ## prune the line by finding the minimum distance
        tgt_dfs=process_xy_back(tx,ty)
        ind=close_list_index(tgt_dfs,data_[start:])
        prune_dfs=tgt_dfs[ind:]

        dfs_save_p.append(prune_dfs)
    ##remove overlap line
    overlap=[]
    for i in range(len(dfs_save_p)):
        for j in range(i,len(dfs_save_p)):
            if i!=j:
                dist=cal_dist(dfs_save_p[i][1],dfs_save_p[j][1])+cal_dist(dfs_save_p[i][-1],dfs_save_p[j][-1])
                # print(dist)
                if dist <0.5:
                    overlap.append(j)
                    break
            else:
                pass
    # print(len(overlap))
    # # print(len(dfs_save_p))
    save_line=[]
    for i in range(len(dfs_save_p)):
        if i not in overlap:
            save_line.append(dfs_save_p[i])

    # np.save(f"{save_dir}/pdfs_{data_name}.npy",np.asarray(dfs_save_p))
    ## search for traj
    input=save_line
    generate_trj=[]
    goal_thre=100
    num_sample=35
    up_to=30 ## how many points for our prediction points
    track_length=3000
    speed=cal_speed_list(data_k[0:19])
    ob_=np.array([[0,0]])
    for j in range(len(input)):
        if len(input[j])>30:
            
            for i in range(num_sample):
                # print(j)
                offset= np.round(np.random.choice(np.linspace(-2,2,9)),2) ## sample the offset from (-lane/2,lane/2)
                # print(offset)
                # offset=2
                prune_dfs_=generate_offset(input[j][0:track_length],offset=offset)
                # print(len(prune_dfs_),len(input[j][0:1000]))



                # prune_dfs_=prune_dfs_new
                

                c_speed=sample_speed(speed[-1])
                
                x1=prune_dfs_[1][0]
                x0=prune_dfs_[0][0]
                y1=prune_dfs_[1][1]
                y0=prune_dfs_[0][1]
                x2=data_[start][0]
                y2=data_[start][1]
                if (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)<0:
                    c_d = -cal_dist(data_[start],prune_dfs_[0]) # current lateral position [m] ## need to get from the current agent's postion offset has problem
                else:
                    c_d = cal_dist(data_[start],prune_dfs_[0])

                if abs(c_d) >4:
                    prune_dfs2=generate_offset(input[j][0:track_length],offset=-offset)
                    prune_dfs_=cal_paral(prune_dfs_=input[j][0:track_length],prune_dfs_2=prune_dfs2)

                x1=prune_dfs_[1][0]
                x0=prune_dfs_[0][0]
                y1=prune_dfs_[1][1]
                y0=prune_dfs_[0][1]
                x2=data_[start][0]
                y2=data_[start][1]

                if (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)<0:
                    c_d = -cal_dist(data_[start],prune_dfs_[0]) # current lateral position [m] ## need to get from the current agent's postion offset has problem
                else:
                    c_d = cal_dist(data_[start],prune_dfs_[0])


                if abs(c_d)>4:
                    pass

                else:

                    c_d=c_d
                    c_d_d = 0.0  # current lateral speed [m/s]
                    c_d_dd = 0.0  # current lateral acceleration [m/s]
                    s0 = 0  # current course position, it means where to start track on the target lane
                    sim_loop=track_length
                    xs=[]
                    ys=[]
                    
                    tx, ty, tyaw, tc, csp = convert_to_frenet(prune_dfs_)
                    for k in range(sim_loop):

                        path = frenet_optimal_planning(
                                    csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob=ob_)
                        if path==None:
                            # print('no path!')
                            break
                        s0 = path.s[1]
                        c_d = path.d[1]
                        c_d_d = path.d_d[1]
                        c_d_dd = path.d_dd[1]
                        c_speed = path.s_d[1]
                        xs=xs+path.x
                        ys=ys+path.y
                        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= goal_thre:
                            # print("Goal")
                            break
                    # pre_x,pre_y=get_direction(process_xy_back(xs,ys))
                    if len(ys) >2: ## if we have find a path that is not none,
                        pre_x,pre_y=np.gradient(xs),np.gradient(ys)
                        # print(pre_x[0],pre_y[0])
                        # d_x,d_y=get_direction(data_)
                        x,y=process_xy(data_)
                        d_x,d_y=np.gradient(x),np.gradient(y)
                        # print(d_x[0],d_y[0])
                        if cal_dist(data_[start],process_xy_back(xs,ys)[0])>10 or (pre_x[0]*d_x[0]<0 and pre_y[0]*d_y[0]<0): ## we dont want the wrong planned traj
                            # generate_trj.append(process_xy_back(xs,ys))
                            pass
                        else:
                            generate_trj.append(inc_length(process_xy_back(xs,ys),up_to=up_to))
    ## save data process here: cut line to the minimum predicted cl:     
    len_line=[]              
    for i in range(len(save_line)):
        len_line.append(len(save_line[i]))
    len_line.sort()
    save_line_=[]
    for i in range(len(save_line)):
        save_line_.append(inc_length(save_line[i],len_line[0]))
    # np.save(f"{save_dir}/trj_{data_name}.npy",np.asarray(generate_trj))
    # print('data',np.asarray(save_line[0],np.float32),np.asarray(generate_trj[0],np.float32))
    return np.asarray(save_line_,np.float32),np.asarray(generate_trj,np.float32) ## change the type as np.float 32? length problem, still remaining need to determine how to prune

def multi_run_wrapper(args):
       return model_generator(*args)

if  __name__ == "__main__":
    ##set root_dir to the correct path to your dataset folder
    root_dir = '/home/qcraft/code/planning-informed-prediction/train_small/data'
    save_dir = '/home/qcraft/code/planning-informed-prediction/save_multi'
    name=os.listdir(root_dir)
    afl = ArgoverseForecastingLoader(root_dir)
    print('Total number of sequences:',len(afl))


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pool = Pool(processes=1)
    pool.map(multi_run_wrapper,[(save_dir,root_dir,name)])
    pool.close()
    pool.join()

    