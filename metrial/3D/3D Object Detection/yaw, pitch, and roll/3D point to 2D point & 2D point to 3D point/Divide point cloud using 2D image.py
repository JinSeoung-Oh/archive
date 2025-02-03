import numpy as np
import torch
class Calibration(object):
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.tag = calibs['tag']
        # Camera intrinsics and extrinsics
        if self.tag == 'reference camera' :
            self.P2 = np.reshape(self.P2, [3,3])
            self.c_u = self.P2[0,2]
            self.c_v = self.P2[1,2]
            self.f_u = self.P2[0,0]
            self.f_v = self.P2[1,1]
        else:
            self.P2 = np.reshape(self.P2, [3,4])
            self.c_u = self.P2[0,2]
            self.c_v = self.P2[1,2]
            self.f_u = self.P2[0,0]
            self.f_v = self.P2[1,1]
            self.b_x = self.P2[0,3] / (-self.f_u)
            self.b_y = self.P2[1,3] / (-self.f_v)
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])
    def read_calib_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        obj = lines[2].strip().split(' ')[1:]
        P2,tag = self.determin_calib(obj)
        #obj = lines[3].strip().split(' ')[1:]
        #P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        return {'P2': P2,
                'tag': tag,
                #'P3': P3.reshape(3, 4),
                'R_rect': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
    def cart2hom(self, pts_3d):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
        return pts_hom
    def determin_calib(self, matrix):
        if len(matrix) == 9:
            matrix = np.array(matrix, dtype=np.float32)
            matrix = matrix.reshape(3,3)
            tag = 'reference camera'
        else:
            if matrix[-1] == 0 and matrix[-2] == 0 and matrix[-3] == 0:
                matrix = np.array(matrix, dtype=np.float32)
                matrix = matrix.reshape(3,3)
                tag = 'reference camera'
            else:
                matrix = np.array(matrix, dtype=np.float32)
                matrix = matrix.reshape(3,4)
                tag = 'not reference  camera'
        return matrix, tag
calib = Calibration('./kitti/test/calib/000000.txt')
 i = calib.P2
e = calib.V2C
#convert 2D coordination to camera coordination
#image corner point have infinite depth, so just fix z vale arbitually value
came = []  
for cor in pts_2d:
    X = cor[0]*(25)
    Y = cor[1]*(25)
    Z = 25
    coor = [[X],[Y],[Z]]
    came.append(coor)
#convert KITTI style intrinsic matrix to general intrinsic matrix format
I = [i[0][0], i[0][1],i[0][2],i[1][0], i[1][1],i[1][2], i[2][0], i[2][1], i[2][2]]
I = np.reshape(I, (3,3))
#convert KITTI style Rotation matrix to general Rotation matrix format
R = [e[0][0], e[0][1],e[0][2],e[1][0], e[1][1],e[1][2], e[2][0], e[2][1], e[2][2]]
R = np.reshape(R, (3,3))
T = [e[0][3], e[1][3], e[2][3]]
T = np.reshape(T, (3,1))
inv_R = np.linalg.inv(R)
inv_I = np.linalg.inv(I)
range_3D = []
for k in range(len(came)):
    came_pts = came[k]
    came_pts = np.array(came_pts)
    range_ = np.matmul(inv_R,np.matmul(inv_I,(came_pts - T)))
    range_3D.append(range_)
re_list = []
for z in range(len(range_3D)):
    pts = range_3D[z]
    point = [[[pts[0]], pts[1]], pts[2]]
    re_list.append(point)
print(re_list)
