import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d import *

from calibration_store import load_stereo_coefficients


def pars():
    # Args handling -> check help parameters to understand
    parser2 = argparse.ArgumentParser(description='Camera calibration')
    parser2.add_argument('--calibration_file', type=str, default='C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\configs\\stereo.yml',
                         help='Path to the stereo calibration file', )
    parser2.add_argument('--left_source', type=str, default='1', help='Left image')
    parser2.add_argument('--right_source', type=str, default='2', help='Right image')
    parser2.add_argument('--pointcloud_dir', type=str, default='C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\stereo2PCL\\output\\',
                         help=' directory path to save pointcloud')
    return parser2.parse_args()


args = pars()
# ====================================
# Get cams params from file "stereo.yml"
K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)

# ======================================
# Factor for downscaling of test images
SCALE = 1
# =====================================
# CAMERA PARAMS
# The focal length of the two cameras, taken from stereo calibration file
FOCAL_LENGTH = Q[2][3]  # FOCAL_L

# The distance between the two cameras, taken from stereo calibration file
X_A = Q[0][3]  # C_X CAMERA L
X_B = K1[0][2]  # C_X CAMERA R
Y = Q[1][3]  # C_Y CAMERA L
DOFFS = X_B - X_A
CAMERA_DISTANCE = 22
# =====================================
# Function declarations
ndisp = 64
vmin = 0
# ====================================
# Undistortion and Rectification part!
scale_percent = 100


# ====================================
# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def dispar_map(imgL, imgR):
    win_size = 7
    min_disp = 2

    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=128,
                                   blockSize=win_size,
                                   uniquenessRatio=8,
                                   speckleWindowSize=0,
                                   speckleRange=2,
                                   disp12MaxDiff=4,
                                   P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=32 * 3 * win_size ** 2,  # 32*3*win_size**2)
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # Compute disparity map
    print("\nComputing the disparity  map...")

    # imgL = cv2.imread('output_l.jpg')
    # imgR = cv2.imread('output_r.jpg')
    global disparity_map
    # gray_left = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    # gray_right = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    disparity_map = stereo.compute(imgL, imgR)

    # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
    # plt.imshow(disparity_map, 'gray')
    # plt.show()

    cv2.imshow('disparity', disparity_map)
    cv2.setMouseCallback("disparity", coords_mouse_disp, disparity_map)
    # Generate  point cloud.
    print("\nGenerating the 3D map...")
    # Get new downsampled width and height
    h, w = imgR.shape[:2]

    # Load focal length.
    focal_length = FOCAL_LENGTH

    # Perspective transformation matrix
    Q2 = Q

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    # cv2.imshow('points_3D', points_3D)
    # Get color points
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()
    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    # Define name for output file
    output_file = 'reconstructed.ply'
    # Generate point cloud
    print("\n Creating the output file... \n")
    create_output(output_points, output_colors, output_file)
    show3d()


def main(parser):
    # cap_l = cv2.VideoCapture('output_L.jpg')
    # cap_r = cv2.VideoCapture('output_R.jpg')

    # cap_l = cv2.imread('output_l.jpg')
    # cap_r = cv2.imread('output_r.jpg')
    # while True:
    # dispar_map(cap_l,cap_r)

    cap_l = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    cap_r = cv2.VideoCapture(2 + cv2.CAP_DSHOW)
    while True:
        ret1, lFrame = cap_l.read()
        ret2, rFrame = cap_r.read()
        dispar_map(lFrame, rFrame)


def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('\nx = ', x, '\ty = ', y)
        Distance = (22 * FOCAL_LENGTH) / (disparity_map[x, y])
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m')


def show3d():
    pcd = o3d.io.read_point_cloud('reconstructed.ply')
    o3d.visualization.draw_geometries([pcd], window_name='PTC', width=640, height=540)
    o3d.visualization.VisualizerWithKeyCallback().update_renderer()
    return


if __name__ == '__main__':
    args = pars()
    main(args)
