import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    qw = quaternion[0]  #q0
    qx = quaternion[1]  #q1
    qy = quaternion[2]  #q2
    qz = quaternion[3]  #q3
    r00 = 2*(qw*qw + qx*qx) -1
    r01 = 2*(qx*qy - qw*qz)
    r02 = 2*(qx*qz + qw*qy)
    r10 = 2*(qx*qy + qw*qz)
    r11 = 2*(qw*qw + qy*qy) -1
    r12 = 2*(qy*qz - qw*qx)
    r20 = 2*(qx*qz - qw*qy)
    r21 = 2*(qy*qz + qw*qx)
    r22 = 2*(qw*qw + qz*qz) -1
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rot_matrix 


"""
neSences calibration format to KITTI calibation format
현재 3D pts to 2D pts & objective angle을 구하기 위해 필요한 calib 정보는 P2(camera intrinsic) / Tr_velo_to_cam / R_rect matrix임

이때, R_rect * Tr_velo_to_cam이 Extrinsic matrix와 대응 됨

따라서, neSences calib 파일에 있는 Extrinsic matrix를 다음과 같이 나타낼 수 있음

     Extrinsic matrix = Rotation matrix * [Tr_velo_to_cam]

     [Tr_velo_to_cam] = (Rotation matrix)^-1 * Extrinsic matrix

    *[Tr_velo_to_cam]은 임의의 matrix이며 이 matrix 값이 KITTI dataset calib 파일의 Tr_velo_to_cam과 

     일치하는 값은 아님

    *(Rotation matrix)^-1은 Rotation matrix의 역행렬을 의미함 

    *R_rect 값에 Rotation matrix 저장
"""
