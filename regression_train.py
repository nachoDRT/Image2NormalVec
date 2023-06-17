from cmath import sin
import os
from re import X
import cv2
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import csv
import pandas as pd
import random

from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Tensorboard
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

torch.set_printoptions(precision=6)


##########################
#        Dataset         #
##########################

IMG_SIZE = 640
XY_IN_MODEL = False
FINE_TUNING = False

TRAIN_NAME = "G1_a"
FOLDER = "29720_imgs"
BST_MODEL = 91
VERSION = "G1_a_Lio_crudo"


class NormalsDataset(Dataset):
    def __init__(
        self,
        data,
        path="./datasets/Regressor_dataset/G1_a",
        type="train",
        point="G1_a_Point_1",
    ):
        # Configuration
        self.path = path
        self.point = point

        # Open labels.txt
        txt_path = os.path.join(path, "labels", point) + "_{}.txt".format(type)
        self.labels = data
        print("Found a total of {} labels for {}".format(len(self.labels), type))

        # Determine image resolution and features and samples
        self.img_size = [IMG_SIZE, IMG_SIZE]
        self.stride = 32
        self.auto = False
        self.n_samples = len(self.labels)
        self.n_features = len(self.labels[0])

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        # Load of image
        # image_path = self.path + '/images/' + \
        #             '{:06d}.png'.format(int(self.labels[index][0]))
        image_path = self.path + "/images/" + self.labels[index][0]
        img0 = cv2.imread(image_path)  # loads on BGR format

        # Aply transformations
        img = self.letterbox(
            img0, new_shape=self.img_size, stride=self.stride, auto=self.auto
        )[0]

        # Convert to tensor
        img = np.ascontiguousarray(img)
        transform = transforms.ToTensor()
        img = transform(img)

        # Load of label
        # TODO Message by Nacho: there seems to be a missunderstanding
        # This was prepared to receive the following format:
        # name.png, x, y, u, v, w
        # Nevertheless, Lio saved:
        # name.png, u, v, w
        # Nacho is going to transform this to work with Lio's way to parse info
        # This might need changes in the future

        # Para lanzar entrenamiento
        # label = torch.FloatTensor(self.labels[index][3:])

        # Para testear
        label = torch.FloatTensor(self.labels[index][1:])
        position = torch.FloatTensor([self.labels[index][1], self.labels[index][2]])
        return img0, img, label, position

    def __len__(self):
        return self.n_samples

    # Resize and pad image while meeting stride-multiple constraints
    def letterbox(
        self,
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Current shape [height, width]
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, ratio, (dw, dh)


##########################
#     MODEL ASSEMBLY     #
##########################


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        message = "".join(
            [
                "c1: ",
                str(c1),
                ", c2: ",
                str(c2),
                ", k: ",
                str(k),
                ", s: ",
                str(s),
                ", p: ",
                str(autopad(k, p)),
            ]
        )
        print(message)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.SiLU()
        else:
            self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class CNNRegression(nn.Module):
    def __init__(self, output_size):
        super(CNNRegression, self).__init__()
        self.conv1 = Conv(3, 6, 1)
        self.conv2 = Conv(6, 16, 1)
        self.conv3 = Conv(16, 64, 1)
        self.conv4 = Conv(64, 128, 1)
        self.conv5 = Conv(128, 256, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, output_size)

    def forward(self, x):
        # -> n, 3, 640, 640
        x = self.pool(self.conv1(x))  # -> n, 6, 320, 320
        x = self.pool(self.conv2(x))  # -> n, 16, 160, 160
        x = self.pool(self.conv3(x))  # -> n, 64, 80, 80
        x = self.pool(self.conv4(x))  # -> n, 128, 40, 40
        x = self.pool(self.conv5(x))  # -> n, 256, 20, 20
        x = x.view(-1, 256 * 20 * 20)  # -> n, 102400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 60
        x = self.fc3(x)  # -> n, 3
        x = F.normalize(x, p=2.0, dim=1)
        return x


class CNNRegressionMaxConv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegressionMaxConv, self).__init__()
        self.conv1 = Conv(3, 6, 1)
        self.conv2 = Conv(6, 16, 1)
        self.conv3 = Conv(16, 32, 1)
        self.conv4 = Conv(32, 64, 1)
        self.conv5 = Conv(64, 128, 1)
        self.conv6 = Conv(128, 256, 1)
        self.conv7 = Conv(256, 512, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, output_size)

    def forward(self, x, saliency: bool = False):
        # -> n, 3, 640, 640
        x = self.pool(self.conv1(x))  # -> n, 6, 320, 320
        x = self.pool(self.conv2(x))  # -> n, 16, 160, 160
        x = self.pool(self.conv3(x))  # -> n, 32, 80, 80
        x = self.pool(self.conv4(x))  # -> n, 64, 40, 40
        x = self.pool(self.conv5(x))  # -> n, 128, 20, 20
        x = self.pool(self.conv6(x))  # -> n, 256, 10, 10
        x = self.pool(self.conv7(x))  # -> n, 512, 5, 5
        x = x.view(-1, 512 * 5 * 5)  # -> n, 512 * 5 * 5
        if saliency:
            saliency_input = x
        x = F.relu(self.fc1(x))  # -> n, 256
        x = F.relu(self.fc2(x))  # -> n, 120
        x = F.relu(self.fc3(x))  # -> n, 60
        x = self.fc4(x)  # -> n, 3
        x = F.normalize(x, p=2.0, dim=1)
        if saliency:
            return x, saliency_input
        else:
            return x


class CNNRegression31OctConv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegression31OctConv, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=17, p=0)
        self.conv2 = Conv(c1=32, c2=64, k=9, p=0)
        self.conv3 = Conv(c1=64, c2=128, k=5, p=0)
        self.pool = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, output_size)

    def forward(self, x, saliency: bool = False):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool2(self.conv3(x))
        x = x.view(-1, 128 * 16 * 16)
        if saliency:
            saliency_input = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2.0, dim=1)
        if saliency:
            return x, saliency_input
        else:
            return x


class CNNRegression224Conv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegression224Conv, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=7, p=2)
        self.conv2 = Conv(c1=32, c2=64, k=5, p=3)
        self.conv3 = Conv(c1=64, c2=128, k=3, p=0)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, output_size)

    def forward(self, x, saliency: bool = False):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool2(self.conv3(x))
        x = x.view(-1, 128 * 18 * 18)
        if saliency:
            saliency_input = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2.0, dim=1)
        if saliency:
            return x, saliency_input
        else:
            return x


class CNNRegression224_xy_Conv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegression224_xy_Conv, self).__init__()
        self.conv1 = Conv(c1=3, c2=32, k=7, p=2)
        self.conv2 = Conv(c1=32, c2=64, k=5, p=3)
        self.conv3 = Conv(c1=64, c2=128, k=3, p=0)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear((128 * 18 * 18) + 2, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, output_size)

    def forward(self, x, labels: torch.float32 = None, saliency: bool = False):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool2(self.conv3(x))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.cat((x, labels), 1)

        # for x_tensor in x:
        #     print(x_tensor)
        if saliency:
            saliency_input = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2.0, dim=1)
        if saliency:
            return x, saliency_input
        else:
            return x


class CNNRegression640_7_Conv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegression640_7_Conv, self).__init__()
        self.output_filters = 512
        self.conv_block_output_size = 3
        self.output_size = output_size
        self.conv1 = Conv(c1=3, c2=8, k=17, p=0)
        self.conv2 = Conv(c1=8, c2=16, k=9, p=0)
        self.conv3 = Conv(c1=16, c2=32, k=5, p=0)
        self.conv4 = Conv(c1=32, c2=64, k=3, p=0)
        self.conv5 = Conv(c1=64, c2=128, k=3, p=1)
        self.conv6 = Conv(c1=128, c2=256, k=3, p=0)
        self.conv7 = Conv(c1=256, c2=self.output_filters, k=3, p=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            self.output_filters
            * self.conv_block_output_size
            * self.conv_block_output_size,
            256,
        )
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, self.output_size)

    def forward(self, x, saliency: bool = False, saliency_regressor: bool = False):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = self.pool(self.conv6(x))
        x = self.pool(self.conv7(x))
        x = x.view(-1, self.output_filters * self.output_size * self.output_size)
        if saliency:
            saliency_input = x
        elif saliency_regressor:
            saliency_input = x
            saliency_to = x
            saliency_to.retain_grad()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2.0, dim=1)
        if saliency:
            return x, saliency_input
        elif saliency_regressor:
            return x, x, saliency_input, saliency_to
        else:
            return x


class CNNRegressionMinConv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        super(CNNRegressionMinConv, self).__init__()
        self.conv1 = Conv(3, 6, 1)
        self.conv2 = Conv(6, 16, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 160 * 160, 60)
        self.fc2 = nn.Linear(60, output_size)

    def forward(self, x):
        # -> n, 3, 640, 640
        x = self.pool(self.conv1(x))  # -> n, 6, 320,320
        x = self.pool(self.conv2(x))  # -> n, 16, 160, 160
        x = x.view(-1, 16 * 160 * 160)  # -> n, 409600
        x = F.relu(self.fc1(x))  # -> n, 60
        x = self.fc2(x)  # -> n, 3
        x = F.normalize(x, p=2.0, dim=1)
        return x


##########################
#       TRAINNING        #
##########################


def progress(epoch, num_epochs, train_loss, val_loss, it_time, percent=0, width=40):
    it_time = time.time() - it_time
    left = width * percent // 100
    right = width - left
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    print(
        "\rEpoch {}/{} -> ".format(epoch, num_epochs - 1),
        "[",
        tags,
        spaces,
        "]",
        " ",
        percents,
        " ",
        sep="",
        end="",
    )
    print("Train Loss: {:.4f}".format(train_loss), end=" ")
    print("Val Loss: {:.4f}".format(val_loss), end=" ", flush=True)
    print("It/s: {:.4f}".format(1 / it_time), flush=True)


##########################
#         MAIN           #
##########################

if __name__ == "__main__":
    TRAIN_THIS_ONE = 9
    PART = "G1_a"
    POINT = 2
    PP = "".join([PART, "_Point_", str(POINT)])

    training_losses = []
    eval_losses = []
    test_losses = []

    #################
    # CONFIGURATION #
    #################

    # point = "G1_a_Point_2"

    # Hyper-parameters
    model_save_path = "./models"
    learning_rate = 1e-3
    num_epochs = 80
    batch_size = 32
    random_seed = 1234
    output_size = 3

    # num_imgs = 15761
    # TODO automate num of images extraction from the dataset
    num_imgs = 29720
    # num_imgs = 306

    dataset_size_trains = [x * num_imgs // 10 for x in range(1, 11)]
    print(f"Starting trainning for {PP}")

    # Tensorboard startup
    board = False
    if board:
        writer = SummaryWriter("runs/regressor")

    # Set up PyTorch configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Setup complete. Using torch {torch.__version__}({device})")
    torch.manual_seed(random_seed)

    # Begin trinings
    for loop_i, dataset_size in enumerate(dataset_size_trains):
        # if loop_i == TRAIN_THIS_ONE:
        # Create directory to save models
        dir = (
            model_save_path + "/" + PART + "/" + PP + "/" + str(dataset_size) + "_imgs"
        )
        try:
            os.makedirs(dir)
        except:
            None

        ###########
        # DATASET #
        ###########
        # Create dataset split
        if FINE_TUNING:
            txt_path = (
                os.path.join("./datasets/Regressor_dataset", VERSION, "labels", PP)
                + ".txt"
            )
        else:
            txt_path = (
                os.path.join("./datasets/Regressor_dataset", PART, "labels", PP)
                + ".txt"
            )

        full_labels = []
        with open(txt_path, "r") as f:
            csvreader = csv.reader(f, delimiter=" ")
            header = next(csvreader)
            for row in csvreader:
                aux = []
                aux.append(row[0])  # Img Name
                aux.append(float(row[1]))  # width
                aux.append(float(row[2]))  # height
                aux.append(float(row[3]))  # u
                aux.append(float(row[4]))  # v
                aux.append(float(row[5]))  # w
                full_labels.append(aux)

        del header, csvreader, f, aux

        # Select random images from the dataset
        rand_list = random.sample(range(len(full_labels)), dataset_size)

        labels = [full_labels[x] for x in rand_list]

        # For several dataset sizes
        # train_val_test, resto = train_test_split(labels,
        #                               test_size=0.99,
        #                               random_state=random_seed,
        #                               shuffle=True)

        # First split of the dataset in train and test
        train_val, test = train_test_split(
            labels, test_size=0.2, random_state=random_seed, shuffle=True
        )

        # Second division in train and validation
        train, val = train_test_split(
            train_val, test_size=0.2, random_state=random_seed, shuffle=True
        )

        test_dict = {
            "Img_name": [item[0] for item in test],
            "x": [item[1] for item in test],
            "y": [item[2] for item in test],
            "u": [item[3] for item in test],
            "v": [item[4] for item in test],
            "w": [item[5] for item in test],
        }

        df = pd.DataFrame(test_dict)
        df.to_csv(dir + "/test_imgs.txt", sep=" ", index=False)

        del labels, train_val, full_labels, rand_list, test_dict, df

        # Creation of dataset and dataloaders
        if FINE_TUNING:
            path = "./datasets/Regressor_dataset/" + VERSION
        else:
            path = "./datasets/Regressor_dataset/" + PART

        datasets = {
            x: NormalsDataset(
                path=path,
                type=x,
                point=PP,
                data=train if x == "train" else val if x == "val" else test,
            )
            for x in ["train", "val", "test"]
        }

        del train, val, test

        dataloaders = {
            x: DataLoader(
                datasets[x], batch_size=batch_size, shuffle=True, num_workers=2
            )
            for x in ["train", "val", "test"]
        }

        dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}

        # Tensorboard
        if board:
            examples_train = iter(dataloaders["train"])
            train_data, train_targets = examples_train.next()
            img_grid = make_grid(train_data)
            writer.add_image("regression_train_images", img_grid)

            examples_val = iter(dataloaders["val"])
            val_data, val_targets = examples_val.next()
            img_grid = make_grid(val_data)
            writer.add_image("regression_val_images", img_grid)

        ##################
        # MODEL ASSEMBLY #
        ##################
        if FINE_TUNING:
            model_name = "".join([PP, "_", str(BST_MODEL), ".pt"])
            model_path = os.path.join("./models", TRAIN_NAME, PP, FOLDER, model_name)

            model = CNNRegression640_7_Conv(3).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            frozen_layers = [
                model.conv1,
                model.conv2,
                model.conv3,
                model.conv4,
                model.conv5,
            ]

            for layer in frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        else:
            model = CNNRegression640_7_Conv(output_size=output_size).to(device)

        # model = CNNRegressionMaxConv(output_size=output_size).to(device)
        # loss and optimizer
        # (ureal-upred)^2 + (vreal-vpred)^2 + (wreal-wpred)^2
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=200, gamma=0.1
        )

        # Tensorboard
        if board:
            writer.add_graph(model, train_data)

        #############
        # TRAINNING #
        #############

        # Training loop dimensions
        total_samples = len(datasets["train"])
        n_iterations = math.ceil(total_samples / batch_size)

        # Evaluation and performance measurement
        since = time.time()
        it_time = time.time()
        best_model_wts = deepcopy(model.state_dict())
        best_loss = float("inf")
        epoch_losses = [[], [], []]
        stop = 0
        print("-" * 96)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                # Iterate over data.
                for i, (original, images, labels, xy_labels) in enumerate(
                    dataloaders[phase]
                ):
                    # Check data integrity
                    original = original[0, :, :, :]
                    original = original.numpy()
                    # cv2.imshow('image', original)
                    # print(labels[0])
                    # cv2.waitKey(0)
                    images = images.to(device)
                    labels = labels.to(device)
                    xy_labels = xy_labels.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        # Forward pass
                        if XY_IN_MODEL:
                            outputs = model(x=images, labels=xy_labels)
                        else:
                            outputs = model(images)
                        loss = criterion(outputs, labels) * batch_size

                        # backward + optimize only if in training phase
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * images.size(0)

                    # Printing of progress
                    if phase == "train":
                        percent = int((i / n_iterations) * 100)
                        train_loss = running_loss / dataset_sizes["train"]
                        if board:
                            global_step = epoch * len(dataloaders["train"]) + i
                            writer.add_scalar("training loss", train_loss, global_step)
                        val_loss = 0.0
                    else:
                        percent = 100
                        val_loss = running_loss / dataset_sizes["val"]
                        if board:
                            global_step = epoch * len(dataloaders["train"]) + i
                            writer.add_scalar("validationloss", val_loss, global_step)

                    progress(
                        epoch,
                        num_epochs,
                        train_loss,
                        val_loss,
                        it_time,
                        percent,
                        width=40,
                    )
                    it_time = time.time()

                # Learning rate scheduler
                if phase == "train":
                    step_lr_scheduler.step()

                # Estimate total epoch loss
                epoch_loss = running_loss / dataset_sizes[phase]

                if phase == "train":
                    epoch_losses[0].append(epoch_loss)
                else:
                    epoch_losses[1].append(epoch_loss)

                # Deep copy the model and early stopping
                if phase == "val":
                    if best_loss - epoch_loss > 0.0001:
                        stop = 0
                        best_loss = epoch_loss
                        best_model_wts = deepcopy(model.state_dict())
                        best_model = model
                        best_epoch = epoch
                        PATH = dir + "/" + PP + "_" + str(epoch) + ".pt"
                        try:
                            torch.save(model.state_dict(), PATH)
                        except:
                            os.makedirs(os.path.join(model_save_path, PART))
                            torch.save(model.state_dict(), PATH)
                        print("New Best model saved")
                    else:
                        stop += 1

            # Calculate the test loss
            with torch.no_grad():
                start_test = time.time()
                running_loss = 0.0
                for i, (original, images, labels, xy_labels) in enumerate(
                    dataloaders["test"]
                ):
                    images = images.to(device)
                    labels = labels.to(device)
                    xy_labels = xy_labels.to(device)
                    if XY_IN_MODEL:
                        outputs = model(x=images, labels=xy_labels)
                    else:
                        outputs = model(images)
                    loss = criterion(outputs, labels) * batch_size
                    running_loss += loss.item() * images.size(0)
                epoch_loss = running_loss / dataset_sizes["test"]
                epoch_losses[2].append(epoch_loss)
                time_test = time.time() - start_test

            if stop == 1000:
                break

            # Carriege return for next epoch
            if epoch < (num_epochs - 1):
                print("\n", end="")

            # Printing of performance statistics
            now = time.time()
            time_elapsed = now - since
            if epoch == 0:
                epoch_time = now - since
            else:
                epoch_time = time_elapsed - time_elapsed_prev
            time_remain = (num_epochs - epoch) * epoch_time
            time_elapsed_prev = time_elapsed
            print("\n", "-" * 96, sep="")
            print(
                "Time since start {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
            print(
                "Epoch time: {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60)
            )
            print(
                "Time remaining: {:.0f}m {:.0f}s".format(
                    time_remain // 60, time_remain % 60
                )
            )
            print("Time test: {:.0f}m {:.0f}s".format(time_test // 60, time_test % 60))
            print("Best model yet: Epoch", str(best_epoch))

        ##############
        # LOG & PLOT #
        ##############
        # Plot train and val error curves
        epochs = range(0, epoch + 1)

        plt.style.use("ggplot")
        plt.plot(epochs, epoch_losses[0], "salmon", label="Training loss")
        plt.plot(epochs, epoch_losses[1], "aquamarine", label="Validation loss")
        plt.plot(epochs, epoch_losses[2], "black", label="Test loss")
        plt.title("Training and Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim([0, 2])
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(dir, PP) + ".png")
        plt.close()

        # Save train and val errors during the train in a csv file
        log_dict = {
            "Train_loss": epoch_losses[0],
            "Val_loss": epoch_losses[1],
            "Test_loss": epoch_losses[2],
        }
        df = pd.DataFrame(log_dict)
        df.to_csv(os.path.join(dir, PP) + ".csv", index=False)

        del log_dict, df

        ##############
        # EVALUATION #
        ##############
        model.load_state_dict(best_model_wts)
        print("Test results:")
        with torch.no_grad():
            acm = np.empty(0, dtype=np.float32)
            total_error = np.empty(0, dtype=np.float32)
            start_test = time.time()
            # Iterate over data.
            for i, (original, images, output, xy_labels) in enumerate(
                dataloaders["test"]
            ):
                images = images.to(device)
                output = output.to(device)
                xy_labels = xy_labels.to(device)

                if XY_IN_MODEL:
                    y_predicted = model(x=images, labels=xy_labels)
                else:
                    y_predicted = model(images)
                error = output - y_predicted
                error = error.abs().sum_to_size(error.shape[1]) / output.shape[0]
                error = error.to("cpu").numpy()
                total_error = np.append(total_error, error, axis=0)
            time_test = time.time() - start_test
            total_error = np.reshape(
                total_error, (total_error.shape[0] // output_size, output_size)
            )
            print(f"\tMean total error = {total_error.mean():.4f}")
            print(f"\tMean u error = {total_error[:,0].mean():.4f}")
            print(f"\tMean v error = {total_error[:,1].mean():.4f}")
            print(f"\tMean w error = {total_error[:,2].mean():.4f}")
            print("Number of predicts: ", str(dataset_sizes["test"]))
            print("Test time: {:.0f}m {:.0f}s".format(time_test // 60, time_test % 60))

        # Record train, val and test min losses for the train
        min_val_loss = min(epoch_losses[1])
        index = epoch_losses[1].index(min_val_loss)
        training_losses.append(epoch_losses[0][index])
        eval_losses.append(min_val_loss)
        test_losses.append(epoch_losses[2][index])

        del model, best_epoch, best_loss, best_model

        print(
            "\n------------------ Train with "
            + str(dataset_size)
            + " images ended ---------------------- \n"
        )

    print(training_losses)
    print(eval_losses)
    print(test_losses)
