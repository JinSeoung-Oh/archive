import math
import os
import sys
import cv2
import numpy as np
import torch
boundary = {
    "minX": -50,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}
bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']
boundary_back = {
    "minX": -50,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}
BEV_WIDTH = 608  # across y axis -25m ~ 25m
BEV_HEIGHT = 608  # across x axis 0m ~ 50m
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT
def makeBEVMap(PointCloud_, boundary):
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1
    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)
    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]
    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height
    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map
    return RGB_Map
def get_lidar(path):
    lidar= np.fromfile(path, dtype=np.float32).reshape(-1,4)
    return lidar
def get_filtered_lidar(lidar, boundary):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']
    mask = np.where((lidar[:,0]>=minX)&(lidar[:,0]<=maxX)&(lidar[:,1]>=minY)&(lidar[:,1]<=maxY)&(lidar[:,2]>=minZ)&(lidar[:,2]<=maxZ))
    lidar = lidar[mask]
    lidar[:,2] = lidar[:,2]-minZ
    return lidar
test_lidar = get_lidar('./007479.bin')
test_lidar = get_filtered_lidar(test_lidar, boundary)
bev_map = makeBEVMap(test_lidar, boundary)
bev_map = torch.from_numpy(bev_map)
bev_map = (bev_map.squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
bev_map = cv2.resize(bev_map, (608,608))
cv2.imwrite('./test.jpg', bev_map)
Open image-20220831-0
