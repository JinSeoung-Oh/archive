import numpy as np
import pandas as pd
from plyfile import PlyData
import open3d as o3d
#아래 코드는 .ply 파일로부터 BEV map을 생성하는 코드
#만약 그냥 x,y,z,r,g,b 값만 가지고 있다면 밑에 datas 부분부터
#run하면 됨
plydata = PlyData.read('./test_result_result.ply')
data = plydata.elements[0].data
data_pd = pd.DataFrame(data)
data_np = np.zeros(data_pd.shape, dtype=np.float)
property_names = data[0].dtype.names
for i, name in enumerate(property_names):
    data_np[:,i] = data_pd[name]
datas=data_np.astype(np.float32)
points = []
colors = []
for x,y,z,r,g,b in datas:
    points.append([x,y,z])
    r = r/255
    g = g/255
    b = b/255
    colors.append([r,g,b])
pts = np.array(points)
pts = pts.reshape(-1,3)
rgb = np.array(colors)
rgb = rgb.reshape(-1,3)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(pts)
pc.colors = o3d.utility.Vector3dVector(rgb)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pc)
vis.get_render_option().point_size = 1.5
vis.run()
vis.capture_screen_image('./testresult.jpg', True)
vis.destroy_window()
Open image-20220829-042217.png
