import os
import pickle
import glob
import cv2

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

models_in_paper = "/Users/vaibhavsingh/Desktop/STFADE/contextnet/contextnet_visualisation/loss_landscape_visualisation/models_used_in_paper"

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
        zaxis_title=z_axis_title),)
    # fig.update_traces(showscale=False)

    fig.update_xaxes(title_font_family="Courier New")
    fig.update_yaxes(title_font_family="Courier New")
    fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=3))

    # fig.show()
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

def make_directories(a):
    current_working_directory_abs = a
    figs_directory_abs = os.path.join(current_working_directory_abs, "figs_normalise")
    try:
        os.mkdir(figs_directory_abs)
        os.mkdir(os.path.join(figs_directory_abs, "log_contour"))
        os.mkdir(os.path.join(figs_directory_abs, "loss_accuracy"))
        os.mkdir(os.path.join(figs_directory_abs, "log_contour_color"))
        os.mkdir(os.path.join(figs_directory_abs, "log_loss_accuracy"))
        os.mkdir(os.path.join(figs_directory_abs, "original_contour"))
        os.mkdir(os.path.join(figs_directory_abs, "original_contour_color"))
        print("directories made successfully!!!!!!", os.getcwd())
    except Exception as e:
        print("-------------Figures directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return figs_directory_abs
    return figs_directory_abs

def normalize(loss_list, global_min, global_max):
    max_n = global_max
    min_n = global_min
    loss_list  = (loss_list - min_n)/(max_n-min_n)
    # print(loss_list)
    return loss_list

def obtain_model_specific_normalisation(loss_lists_directory):
    global_max = -1
    global_min = 1000000
    for i, f in enumerate(tqdm(sorted(os.listdir(loss_lists_directory)))):
        if f == ".DS_Store":
            continue
        ff = os.path.join(loss_lists_directory, f)
        if os.path.isdir(ff):
            continue
        print("For normalisaiton filename is  ", ff)
        with open(ff, "rb") as model_file:
            x_temp = pickle.load(model_file)

        loss_list = x_temp['loss_list'][0]
        global_max = max(global_max, loss_list.max())
        global_min = min(global_min, loss_list.min())

    print("the global minima and maxima is ", global_min, global_max)
    return global_max, global_min

global_max, global_min = obtain_model_specific_normalisation(models_in_paper)
figs_directory_abs = make_directories(models_in_paper)

for i, file in enumerate(tqdm(sorted(os.listdir(models_in_paper)))):
    break
    if file == ".DS_Store":
        continue
    model_file_list = os.path.join(models_in_paper, file)
    if not os.path.isdir(model_file_list):
        continue
    print("file to be opened for proecessing ", )
    with open(model_file_list, "rb") as model_file:
        x_temp = pickle.load(model_file)

    loss_list = x_temp['loss_list'][0]
    acc_list_greedy_char = x_temp['greedy_char'][0]
    acc_list_greedy_wer = x_temp['greedy_wer'][0]
    acc_list_beam_wer = x_temp['beam_wer'][0]
    acc_list_beam_char = x_temp['beam_char'][0]
    acc_list = acc_list_beam_char
    loss_list = normalize(loss_list, global_min, global_max)
    create_viz(loss_list, acc_list_beam_char, figs_directory_abs, file)

figures_working_dir = os.path.join(models_in_paper, "figs_normalise")
video_directory = os.path.join(os.getcwd(), "video")
try:
    mkdir(video_directory)
    print("video dir created successfully", video_directory)
except:
    print("folder already exists")
    print(video_directory)
fname1 = figures_working_dir+'/loss_accuracy/*.png'
fname2 = figures_working_dir+'/original_contour/*.png'

index = 1

size = (10,10)
img_array = []
for filename1, filename2 in zip(sorted(glob.glob(fname2)), sorted(glob.glob(fname1))):
    print(filename1)
    print(filename2)

    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
    height, width, layers = image1.shape
    size = (width, height)
    print(size)
    height, width, layers = image2.shape
    size = (width, height)
    print(size)

    vis = cv2.hconcat([image1, image2])
    height, width, layers = vis.shape
    size = (width, height)

    font = cv2.QT_FONT_NORMAL
    # org
    org = (460, 28)

    # fontScale
    fontScale = 0.8

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(vis, 'Epoch ' + str(index), org, font,
                        fontScale, color, thickness, cv2.LINE_4)

    img_array.append(image)
    index = index + 1

filename = video_directory + "/loss_video.mp4"
print(filename)
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()




