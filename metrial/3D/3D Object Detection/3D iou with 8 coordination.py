import numpy as np
from scipy.spatial import ConvexHull
from numpy import *
#Ref : https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
#Sutherland-Hodgman_polygon_clipping 알고리즘은 arbitrary polygon (subjectPolygon)과 
#convex polygon (clipPolygon) 사이의 intersection polygon을 찾아 내는 알고리즘
#아래의 코드는 python으로 구현 되어 있는 코드
def polygon_clip(subjectPolygon, clipPolygon):
    def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)
def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  
def box3d_vol(corners):
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c
def box3d_iou(corners1, corners2):
    # corner points are in counter clockwise order
    # 각각의 coner points들은 xz 평면으로 정사형(orthogonal projection)한 point 좌표들임
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d
(IOU_3d,IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)

-------------------------------------------------------------------------

from pytorch3d.ops import box3d_overlap
# Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
intersection_vol, iou_3d = box3d_overal(boxes1, boxes2)
