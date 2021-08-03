# Copyright 2021 Vaibhav Singh (@vaibhav016)
# Copyright 2021 Dr Vinayak Abrol (_)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objs as go
from tqdm import tqdm

number_of_points = 9
small_range = -1.0
large_range = 1.0

xcoordinates = np.linspace(small_range, large_range, num=number_of_points)
ycoordinates = np.linspace(small_range, large_range, num=number_of_points)

xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)

loss_lists_directory = os.path.join(os.getcwd(), "loss_lists")


def plot_fig(loss_list, log_=False):
    if log_ == True:
        loss_list = np.log(loss_list)
        l, u, d = 3, 12, 1
    else:
        l, u, d = 0, 1, 0.1

    data = [
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh, z=loss_list, colorscale='Jet', cmin=0, cmax=u, opacity=0.9,
            contours=go.surface.Contours(z=go.surface.contours.Z(show=True, usecolormap=True, project=dict(z=True), ),
                                         )
        )
    ]

    layout = go.Layout(autosize=False,
                       scene=dict(dict(
                           xaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(163, 221, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New"),
                           yaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(91, 122, 133)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New"),
                           zaxis=dict(range=[l, u],
                                      backgroundcolor="rgb(204, 231, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=1, dtick=d, title_font_family="Courier New")),
                           camera=dict(eye=dict(x=2, y=5, z=1.5))),

                       margin=dict(l=30, r=10, b=20, t=10),
                       width=500, height=500)
    fig = go.Figure(data=data, layout=layout)
    if log_:
        z_axis_title = "Log loss"
    else:
        z_axis_title = "loss"

    fig.update_layout(scene=dict(
        xaxis_title='λ',
        yaxis_title='η',
        zaxis_title=z_axis_title), )

    fig.update_xaxes(title_font_family="Courier New")
    fig.update_yaxes(title_font_family="Courier New")
    fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=3))

    return fig


def create_viz(loss_list, acc_list, figure_directory, filename, title="none"):
    print(filename)
    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title(title)
    CS = plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/original_contour/" + filename + "OriginalContour.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title(title)
    plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, fontsize=8, inline=1, fmt='%2.1f')
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.colorbar(CS)
    plt.savefig(figure_directory + "/original_contour_color/" + filename + "OriginalContourColor.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title(title)
    CS = plt.contour(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/log_contour/" + filename + "LogScale.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title(title)
    plt.contour(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, fontsize=8, inline=1, fmt='%2.1f')
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/log_contour_color/" + filename + "LogScale.png")

    fig_loss_only = plot_fig(loss_list)
    fig_loss_log = plot_fig(loss_list, log_=True)

    fig_loss_only.write_image(figure_directory + "/loss_accuracy/" + filename + "Loss_Accuracy.png")
    fig_loss_log.write_image(figure_directory + "/log_loss_accuracy/" + filename + "Log_Loss_Accuracy.png")


def make_directories():
    current_working_directory_abs = os.getcwd()
    figs_directory_abs = os.path.join(current_working_directory_abs, "figs")
    try:
        os.mkdir(figs_directory_abs)
        os.mkdir(os.path.join(figs_directory_abs, "log_contour"))
        os.mkdir(os.path.join(figs_directory_abs, "loss_accuracy"))
        os.mkdir(os.path.join(figs_directory_abs, "log_contour_color"))
        os.mkdir(os.path.join(figs_directory_abs, "log_loss_accuracy"))
        os.mkdir(os.path.join(figs_directory_abs, "original_contour"))
        os.mkdir(os.path.join(figs_directory_abs, "original_contour_color"))
        print(os.getcwd())
    except Exception as e:
        print("-------------Figures directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return figs_directory_abs
    return figs_directory_abs


def normalize(loss_list, global_min, global_max):
    max_n = global_max
    min_n = global_min
    loss_list = (loss_list - min_n) / (max_n - min_n)
    # print(loss_list)
    return loss_list

figs_directory_abs = make_directories()

def plot_loss_landscape(figs_directory_abs):
    for file in tqdm(sorted(os.listdir(loss_lists_directory))):
        filename = os.path.join(loss_lists_directory, file)
        print("filenmes ", filename)
        with open(filename, "rb") as f:
            x_temp = pickle.load(f)
        file = file.split('.')[0]
        loss_list = x_temp['loss_list'][0]
        acc_list_greedy_char = x_temp['greedy_char'][0]
        acc_list_greedy_wer = x_temp['greedy_wer'][0]
        acc_list_beam_wer = x_temp['beam_wer'][0]
        acc_list_beam_char = x_temp['beam_char'][0]
        acc_list = acc_list_beam_char

        create_viz(loss_list, acc_list_beam_char, figs_directory_abs, file)


if __name__ == '__main__':
    plot_loss_landscape(figs_directory_abs)
