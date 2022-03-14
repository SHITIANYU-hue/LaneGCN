#!/usr/bin/env python

"""A simple python script template."""
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from utils.others import process_xy

from argoverse.map_representation.map_api import ArgoverseMap

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}





def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def viz_sequence(
    df: pd.DataFrame,
    predict_cl:Optional[List[np.ndarray]] = None,    
    predict_tj:Optional[List[np.ndarray]] = None,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = True,
    show_predict_cl: bool= False,
    show_predict_tj: bool= False,

    smoothen: bool = False,
) -> None:

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    plt.figure(0, figsize=(8, 7))

    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "--",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    frames = df.groupby("TRACK_ID")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "red", "OTHERS": "gray", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)
    # plot the prediction result:
    if show_predict_cl:
        for i in range(len(predict_cl)):
            x,y=process_xy(predict_cl[i])
            plt.plot(x,y,"-",color='gold',linewidth=1,label='cl track')

    if show_predict_tj:
        for i in range(len(predict_tj)):
            x,y=process_xy(predict_tj[i])
            plt.plot(x,y,"o--",color='blue',linewidth=0.2,markersize=0.2,label='predict traj')

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        if object_type =="AGENT":
            plt.plot(
                cor_x[0:20],
                cor_y[0:20],
                "-",
                color='purple',
                label='agent track(obs) ',
                alpha=1,
                linewidth=1,
                zorder=_ZORDER[object_type],
            )         

            plt.plot(
                cor_x[20:],
                cor_y[20:],
                "-",
                color='red',
                label='agent gt',
                alpha=1,
                linewidth=1,
                zorder=_ZORDER[object_type],
            )       

        else:
            plt.plot(
                cor_x,
                cor_y,
                "-",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=1,
                zorder=_ZORDER[object_type],
            )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 4
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7

        plt.plot(
            final_x,
            final_y,
            marker_type,
            color=color_dict[object_type],
            # label=object_type if not object_type_tracker[object_type] else "", it seems that we don't need it
            alpha=1,
            markersize=marker_size,
            zorder=_ZORDER[object_type],
        )

        object_type_tracker[object_type] += 1

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=7,
        label="Others",
    )
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")

    plt.axis("off")
    if show:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()



