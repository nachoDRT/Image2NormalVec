""" python object_detector-main/test_regresor.py
"""

from importlib.resources import path
from pickle import TRUE
from pickletools import uint8
from turtle import color
from unicodedata import name
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from regression_train import CNNRegression
from regression_train import CNNRegressionMaxConv
from regression_train import CNNRegressionMinConv
from regression_train import CNNRegression31OctConv
from regression_train import CNNRegression224Conv
from regression_train import CNNRegression640_7_Conv
from regression_train import NormalsDataset
from glob import glob
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import csv
from sklearn.model_selection import train_test_split
import math
import numpy as np
from pandas import *
import seaborn as sb
import shutil

# Model data
NUM_LAYERS = 7
OLD_MODEL = False
# TRAIN_NAME = "G1_a_complete_M7_UP"
TRAIN_NAME = "G1_a"
FOLDER = "306_imgs"
BST_MODEL = 45

# Other data
DATASET_BATCHES = 10

FONT_SIZE = 10

# Analysis
TEST_PLOTS = True
SHOW_VECTORS = False
DSET_BATCHES = False

SALIENCY = False
SALIENCY_REGRESSOR = True
SHOW_SALIENCY = False

PART = "G1_a"
POINT = 2
PP = "".join([PART, "_Point_", str(POINT)])
# VERSION = "_test_individual_parts"
VERSION = "_Lio"
# VERSION = "_saliency"
# VERSION = ""

INTERVALS = 10
STEP = 1 / INTERVALS
BOXES = 5

ANGLE_LIMIT = 20

PATH = os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, FOLDER)

IMG_FONT = cv2.FONT_HERSHEY_SIMPLEX
# IMG_SCALE = 0.8
IMG_SCALE = 0.35

# Violin plots
MAIN_FONT = 26
MEDIUM_FONT = 16
SMALL_FONT = 12
TINY_FONT = 8

XY_IN_MODEL = False


def save_suspicius_image(*, good_img: bool = False, delete=False, index=None, img=None,
                         img_name=None, label=None, prediction=None):

    """Create folders for the different bins to store those predictions that are wrong
    """

    # Parent folder
    if good_img:
        parent_folder = os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, "good_imgs")
    else:
        parent_folder = os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, "suspicious_imgs")
    if delete:
        if os.path.exists(parent_folder):
            shutil.rmtree(parent_folder)

    else:
        if not os.path.isdir(parent_folder):
            os.mkdir(parent_folder)

        a = round(0.1 * index, 1)
        b = round(a + 0.1, 1)
        projection = "".join([str(a), "_to_", str(b), "_projection"])

        child_folder = os.path.join(parent_folder, ''.join([projection, '_errors']))
        if not os.path.isdir(child_folder):
            os.mkdir(child_folder)

        label = [round(item, 2) for item in label]
        prediction = [round(item, 2) for item in prediction]

        img_path = os.path.join(child_folder, img_name)
        img = cv2.putText(img, "".join(["Label:      ", str(label)]), org=(20, 20),
                          fontFace=IMG_FONT, fontScale=IMG_SCALE, color=(0, 255, 0))
        img = cv2.putText(img, "".join(["Prediction: ", str(prediction)]), org=(20, 40),
                          fontFace=IMG_FONT, fontScale=IMG_SCALE, color=(0, 0, 255))

        cv2.imwrite(filename=img_path, img=img)


def load_architecture(n_layers):
    """ Load proper architecture as model
    """
    regresor = False
    if n_layers == 3:
        if OLD_MODEL:
            regresor = CNNRegressionMinConv(3).to(device)
        else:
            regresor = CNNRegression224Conv(3).to(device)
    elif n_layers == 5:
        regresor = CNNRegression(3).to(device)
    elif n_layers == 7:
        if OLD_MODEL:
            regresor = CNNRegressionMaxConv(3).to(device)
        else:
            regresor = CNNRegression640_7_Conv(3).to(device)
    else:
        print("Error. Wrong architecture selection")
    return regresor


def dset_batches():
    """ Generate a plot showing the 'validation error' vs. 'dataset size', i.e.:
    The dataset is 'n' images and we train the model 'N' times.
    We train the model adding 'n/10' images every time.
    """

    # Dictionary to store errors & dataset fraction
    val_dictionary = {}

    path = "./models/" + TRAIN_NAME + "/G1_a_Point_2"

    for filename in os.listdir(path):
        # Extract dataset fraction
        dset_fraction = filename[:-5]

        # Open '.csv' whith stored errors
        filename += "/G1_a_Point_2.csv"
        filename = os.path.join(path, filename)
        data = read_csv(filename)
        val_loss = data['Val_loss'].tolist()
        val_loss.sort()

        # Save smallest validation error in dictionary
        val_dictionary[dset_fraction] = val_loss[:1][0]

    # Arrange the dictionary (ascending order)
    val_dictionary = dict(sorted(val_dictionary.items(), key=lambda item: int(item[0])))

    # Take sorted validation errors
    val_errors = val_dictionary.values()

    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Dataset %', fontsize=14)
    ax.set_ylabel('Min. Validation error', fontsize=14)
    ax.set_title("Convolutional Layers: " + str(NUM_LAYERS))

    dset_batches = range(10, (DATASET_BATCHES + 1)*10, 10)
    for i, key in enumerate(val_dictionary.keys()):
        print(dset_batches[i], val_dictionary[key])
        ax.annotate('n = ' + key, (dset_batches[i], val_dictionary[key]), xytext=(-20, 10), textcoords='offset points')

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlim([-1, 105])
    plt.scatter(dset_batches, val_errors)
    plt.show()


def progress(percent=0, width=40):
    left = width * percent // 100
    right = width - left
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    print("[", tags, spaces, "]", " ", percents, " ", sep="", flush=True)
    # print('Train Loss: {:.4f}'.format(train_loss), end=' ')
    # print('Val Loss: {:.4f}'.format(val_loss), end='', flush=True)


def get_coordinate(mat):
    x = int(mat[0][0].item()*640)
    y = int(mat[0][1].item()*640)
    return (x, y)


def get_projection(mat):
    x = int(mat[0][0].item() * 640 + mat[0][3].item()*100)
    y = int(mat[0][1].item() * 640 + mat[0][2].item()*100)
    print(mat[0][1].item() * 640, mat[0][3].item()*100)
    print(mat[0][0].item() * 640, mat[0][2].item()*100)
    print("projection: "+str((x, y)))
    return (x, y)


def press(event):
    if event.key == 'q':
        return


def save_figure(boxplot, name, xticks_data):

    scatter_list = []

    plt.clf()
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('Bins [-]')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_title('')
    boxplot_values = ax.boxplot(boxplot)
    if name == "angles" and VERSION == "" or VERSION == "_Lio" or VERSION == "_Lio_crudo" or VERSION == "_test_individual_parts":
        scatter_list = [round(item.get_ydata()[1], 1) for item in boxplot_values['medians']]
        ax.set_ylabel("".join([name, " [º]"]))
        plt.ylim([0, 60])
    else:
        scatter_list = [round(item.get_ydata()[1], 2) for item in boxplot_values['medians']]
        ax.set_ylabel("".join([name, " [-]"]))
        plt.ylim([0, 2])

    unique_bins, bins_counter = np.unique(xticks_data, return_counts=True)
    lenght = len(unique_bins)
    for i, _ in enumerate(boxplot):
        if i >= lenght:
            unique_bins = np.append(0, unique_bins)
            bins_counter = np.append(0, bins_counter)

    xticks_labels = ["".join([str(round(float(bin) * 0.1, 1)), ' - ', str(round(float(bin) * 0.1 + 0.1, 1)),
                             "\n", str(bins_counter[i]), " samples"]) for i, bin in enumerate(unique_bins)]

    xticks = plt.xticks()
    plt.xticks(xticks[0], labels=xticks_labels, fontsize=SMALL_FONT)
    plt.savefig(os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, FOLDER,
                "".join([name, VERSION, '.png'])))

    plt.close()
    return scatter_list, bins_counter


def save_scatter_plot(*, y_value: list, x_value: list, name: str):
    plt.clf()
    plt.style.use('ggplot')
    plt.scatter(x=x_value, y=y_value)
    plt.title("".join(['Error vs. Dataset size (', name, ")"]), fontsize=MEDIUM_FONT, loc='left')
    if name == "angles":
        plt.ylabel("".join(['Median', " [º]"]), fontsize=MEDIUM_FONT)
    else:
        plt.ylabel("".join(['Median', " [-]"]), fontsize=MEDIUM_FONT)
    plt.xlabel('Number os samples', fontsize=MEDIUM_FONT)
    plt.savefig(os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, FOLDER,
                "".join(["median_scatterplot_", name, VERSION, '.png'])))
    plt.close()


def save_violin(*, classes, values):
    plt.clf()
    sb.set(style="whitegrid", palette="pastel", color_codes=True)
    plt.subplots(1, figsize=(16, 8), facecolor="white")
    sb.violinplot(x=classes, y=values, split=False, inner="quart", color="white")

    plt.title('Error distribution', fontsize=MAIN_FONT, loc='left')
    plt.legend(frameon=False, fontsize=MEDIUM_FONT, loc='upper left', ncol=2, bbox_to_anchor=(0.7, 1.1))
    plt.ylabel('Angle/Distance', fontsize=MEDIUM_FONT)
    plt.xlabel('Vectors Dot Product', fontsize=MEDIUM_FONT)
    xticks = plt.xticks()

    unique_classes, num_samples = np.unique(classes, return_counts=True)
    xticks_labels = ["".join([str(round(float(bin) * 0.1, 1)), ' - ', str(round(float(bin) * 0.1 + 0.1, 1)),
                     "\n", str(num_samples[i]), " samples"]) for i, bin in enumerate(unique_classes)]

    plt.xticks(xticks[0], labels=xticks_labels, fontsize=SMALL_FONT)
    plt.yticks(np.arange(0, 110, 20), fontsize=SMALL_FONT)
    # Footer
    note = '*Cátedra de Industria Conectada'
    source = '\nInstitute for Research in Technology'
    plt.text(10.5, -24, note+source, ha='right', fontsize=TINY_FONT)
    plt.savefig(os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, FOLDER,
                "".join(['violin_plot', VERSION, '.png'])))


def add_og_and_map(*, og, heat_map, delete: bool = False, name):
    rows, cols = og.shape[:2]
    roi = og[0:rows+0, 0:cols+0]
    # heat_map = heat_map.swapaxes(0, 2)

    gray_img = cv2.cvtColor(heat_map, cv2.COLOR_BGR2GRAY)
    gray_og = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)

    backtorgb = cv2.cvtColor(gray_og, cv2.COLOR_GRAY2RGB)
    backtorgb = np.uint16(backtorgb)

    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    base_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    heat_map = np.uint8(heat_map)
    mask = np.uint8(mask)

    img_fg = cv2.bitwise_and(src1=heat_map, src2=heat_map, mask=mask)

    img_fg = np.float32(img_fg)
    img_fg = img_fg/255

    dst = cv2.add(base_img_bg, img_fg)
    og[0:rows+0, 0:cols+0] = dst

    parent_folder = os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, "saliency_imgs")
    if delete:
        if os.path.exists(parent_folder):
            shutil.rmtree(parent_folder)
    if not os.path.isdir(parent_folder):
        os.mkdir(parent_folder)

    cv2.imwrite(os.path.join(parent_folder, name), og*255)
    if SHOW_SALIENCY:
        cv2.imshow("img", og)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return og


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    criterion = nn.MSELoss()
    batch_size = 32
    # Load the model to test
    regresor = load_architecture(NUM_LAYERS)
    model = "".join([PP, "_", str(BST_MODEL), ".pt"])
    model_path = os.path.join("./models", TRAIN_NAME, PP, FOLDER, model)
    regresor.load_state_dict(torch.load(model_path, map_location=device))

    regresor.eval()

    # filters = regresor.conv1.weights.detach().clone()
    # filgers = regresor.conv1.weight
    # print(filters)
    # print(a)

    # imgs_path = os.path.join(os.getcwd(), 'datasets', 'Regressor_dataset', PART, 'images')
    # txt_path = os.path.join(os.getcwd(), 'models', TRAIN_NAME, PP, FOLDER, 'test_imgs.txt')

    # imgs_path = '/home/msi/Documentos/database_generator/
    #              G1_a_flatten_dataset/box_0.8_to_0.9_projection/dataset_4_regressor/images'
    imgs_path = os.path.join(os.getcwd(), 'datasets', 'Regressor_dataset', "".join([PART, VERSION]), 'images')

    if VERSION == "":
        txt_path = os.path.join(os.getcwd(), 'models', TRAIN_NAME, PP, FOLDER, 'test_imgs.txt')

    else:
        txt_path = os.path.join(os.getcwd(), 'datasets', 'Regressor_dataset',
                                "".join([PART, VERSION]), 'labels', "".join([PP, '.txt']))

    # Remove previous suspicious images folde (images linked to great angle errors)
    save_suspicius_image(delete=True)

    labels_txt = []
    with open(txt_path, 'r') as f:
        csvreader = csv.reader(f, delimiter=' ')
        header = next(csvreader)
        for row in csvreader:
            aux = []
            aux.append(row[0])
            # aux.append(float(row[1]))
            # aux.append(float(row[2]))
            # aux.append(float(row[3]))
            aux.append(float(row[3]))
            aux.append(float(row[4]))
            aux.append(float(row[5]))
            labels_txt.append(aux)

    test = labels_txt

    n_iterations = len(test)

    del header, csvreader, f, aux

    datasets = {'test': NormalsDataset(path='./datasets/Regressor_dataset/' + "".join([PART, VERSION]),
                                       type='test',
                                       point=PP,
                                       data=test)}

    dataloaders = {'test': DataLoader(datasets['test'])}

    if TEST_PLOTS:
        boxplot_vec = [[], [], [], [], [], [], [], [], [], []]
        boxplot_ang = [[], [], [], [], [], [], [], [], [], []]
    running_loss = 0
    counter = 0

    bins_list = []
    angles = []
    sal_folder_flag = True
    for i, (original, img, lbl, extra) in enumerate(dataloaders['test']):
        images = img.to(device)
        labels = lbl.to(device)
        extra = extra.to(device)

        if SALIENCY:

            for param in regresor.parameters():
                param.requires_grad = False

            images.requires_grad = True

            if XY_IN_MODEL:
                outputs, saliency_inputs = regresor(x=images, labels=extra, saliency=True)
            else:
                outputs, saliency_inputs = regresor(x=images, saliency=True)
            loss = criterion(outputs, labels) * batch_size
            running_loss += loss.item()*images.size(0)

            saliency_inputs.sum().backward()

            saliency, _ = torch.max(torch.abs(images.grad[0]), dim=0)

            saliency_host = saliency.cpu().numpy()
            saliency_host = (saliency_host - saliency_host.min()) * 255 / (saliency_host.max() - saliency_host.min())
            image_host = images.cpu().detach().numpy()
            image_host = image_host[0].transpose(1, 2, 0)

            saliency_host = (saliency_host).astype(np.uint8)

            heatmap = cv2.applyColorMap(saliency_host, cv2.COLORMAP_HOT)
            og_and_map = add_og_and_map(og=image_host, heat_map=heatmap, delete=sal_folder_flag, name=labels_txt[i][0])
            sal_folder_flag = False

        elif SALIENCY_REGRESSOR:
            for param in regresor.parameters():
                param.requires_grad = False

            images.requires_grad = True

            outputs, saliency_inputs = regresor(x=images, saliency_regressor=True)
        else:
            with torch.no_grad():
                activation = {}
                activation['conv1'] = outputs.detach()
                regresor.conv1.register_forward_hook(activation['conv1'])

                outputs = regresor(images)
                loss = criterion(outputs, labels) * batch_size
                running_loss += loss.item()*images.size(0)

                act = activation['conv1'].squeeze()
                act_host = act.cpu().numpy()
                fig, ax = plt.subplots(act.size(0))
                for idx in range(act.size(0)):
                    print(act_host[idx].shape)
                    ax[idx].imshow(act_host[idx])
                plt.show()
                plt.close()

        percent = int((i/n_iterations)*100)
        progress(percent, width=40)
        original = original[0, :, :, :]
        original = original.numpy()

        # soa = np.array([[0, 0, 0, labels[0][0].item(), labels[0][1].item(), labels[0][2].item()],
        #                [0, 0, 0, outputs[0][0].item(), outputs[0][1].item(), outputs[0][2].item()]])
        # v_label = []
        # X, Y, Z, U, V, W = zip(*soa)
        if SHOW_VECTORS:

            img_path = "".join([imgs_path, labels_txt[i][0]])
            img = cv2.imread(img_path)
            cv2.imshow('image', img)
            cv2.waitKey(0)

            fig = plt.figure(labels_txt[i][0])
            fig.canvas.mpl_connect('key_press_event', press)
            ax = fig.add_subplot(111, projection='3d')

            ax.quiver(0, 0, 0, labels[0][0].item(), labels[0][1].item(), labels[0][2].item(), color='blue')
            ax.quiver(0, 0, 0, outputs[0][0].item(), outputs[0][1].item(), outputs[0][2].item(), color='red')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            plt.legend(["Label", "Output"])
            plt.show()

        if TEST_PLOTS:
            # Store both vectors
            # print(labels)
            labelled = [labels[0][0].item(), labels[0][1].item(), labels[0][2].item()]
            predicted = [outputs[0][0].item(), outputs[0][1].item(), outputs[0][2].item()]

            # Store z value
            z_labelled = labels[0][2].item()
            index = math.trunc(z_labelled*10)
            if index == 10:
                index = 9

            # Compute difference
            eDistance = math.dist(labelled, predicted)
            dot_product = np.dot(labelled, predicted)
            angle = np.arccos(dot_product)

            if np.isnan(angle):
                angle = 0

            angle = math.degrees(angle)
            if angle > ANGLE_LIMIT:
                img_path = os.path.join(imgs_path, labels_txt[i][0])
                img = cv2.imread(img_path)
                save_suspicius_image(index=index, img=img, img_name=labels_txt[i][0],
                                     label=labelled, prediction=predicted)
            else:
                img_path = os.path.join(imgs_path, labels_txt[i][0])
                img = cv2.imread(img_path)
                save_suspicius_image(good_img=True, index=index, img=img, img_name=labels_txt[i][0],
                                     label=labelled, prediction=predicted)
            # Save it
            boxplot_vec[index].append(eDistance)
            boxplot_ang[index].append(angle)

            bins_list.append(index)
            angles.append(angle)
        counter += 1
    if TEST_PLOTS:
        vect_scatter_list, _ = save_figure(boxplot=boxplot_vec, name='diff_vector', xticks_data=bins_list)
        angl_scatter_list, bins_counter = save_figure(boxplot=boxplot_ang, name='angles', xticks_data=bins_list)
        save_scatter_plot(y_value=vect_scatter_list, x_value=bins_counter, name='diff_vector')
        save_scatter_plot(y_value=angl_scatter_list, x_value=bins_counter, name='angles')

        save_violin(classes=bins_list, values=angles)

        # fig = plt.figure(figsize=(10, 7))
        # # Creating axes instance
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # # Creating plot
        # bp = ax.boxplot(boxplot_vec)
        # ax = plt.gca()  # get current axis object
        # ax.annotate('n_samples: ' + str(len(boxplot_vec[9])), xy=(.5, .5),  xycoords='axes fraction',
        #             xytext=(0.2, 0.95), textcoords='axes fraction')
        # # show plot
        # plt.show()
        # plt.savefig(os.path.join(os.getcwd(), "models", TRAIN_NAME, PP, FOLDER, "distances.png"))
        # plt.clf()

    if DSET_BATCHES:
        dset_batches()

    cv2.destroyAllWindows()
    # cv2.arrowedLine(original, (320,320), (320+int(100*labels[0][0].item()),320+int(100*labels[0][1].item())),
    #                               (255,0,0), 5)
    # cv2.arrowedLine(original, (320,320), (320+int(100*outputs[0][0].item()),320+int(100*outputs[0][1].item())),
    #                               (0,0,255), 5)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(original, str([round(labels[0][0].item(),2),round(labels[0][1].item(),2),
    #   round(labels[0][2].item(),2)]), (50,50), font,0.7,(100,100,0))
    # cv2.putText(original, str([round(outputs[0][0].item(),2),round(outputs[0][1].item(),2),
    # round(outputs[0][2].item(),2)]), (50,100), font,0.7,(100,100,0))
    # cv2.imshow('Original', original)
    # # Press Q to exit
    # if cv2.waitKey(0) == ord('q'):
    #     continue
    print("Average loss = " + str(running_loss/len(test)))
    # cv2.destroyAllWindows()
