from random import randint, seed
import numpy as np

import math
import pandas as pd
from scipy.optimize import linear_sum_assignment
class SearchEntry():
    

    def __init__(self, x, y,z,t, g_cost, f_cost=0, pre_entry=None): # z cooridinate
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.pre_entry = pre_entry

    def getPos(self):
        return (self.x, self.y,self.z,self.t)

    def __str__(self):
        return 'x = {}, y = {}, z = {}, t = {}, f = {}'.format(self.x, self.y, self.z, self.t, self.f_cost)
    def __lt__(self, other):
        """重载 < 运算符，使 Python 可以比较 f_cost"""
        return self.f_cost < other.f_cost

    def __gt__(self, other):
        """重载 > 运算符"""
        return self.f_cost > other.f_cost

class Map():

    def __init__(self, width, height,Zlength,Time,volume_x,volume_y,volume_z):
        self.width = width
        self.height = height
        self.Zlength = Zlength
        self.Time= Time
        self.volume_x=volume_x
        self.volume_y=volume_y
        self.volume_z=volume_z

        self.map = np.zeros((Time, Zlength, height, width), dtype=int)

    def createBlock(self, paths):
 
        tt=0
        for i, path in enumerate(paths):
            
            x=path[0]
            y=path[1]
            z=path[2]
            x_min, x_max = max(x - self.volume_x, 0), min(x + self.volume_x, map.width - 1)
            y_min, y_max = max(y -  self.volume_y, 0), min(y +  self.volume_y, map.height - 1)
            z_min, z_max = max(z - self.volume_z, 0), min(z + self.volume_z, map.Zlength - 1)
            if i == len(paths) - 1:
                self.map[tt:self.Time+1, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1
            else:
                self.map[tt:tt+2, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1
                #tt:tt+3 是左闭右开所以总共是标2帧
            tt=tt+1

    def generatePos(self, rangeX, rangeY,rangez):
        x, y,z = (randint(rangeX[0], rangeX[1]), randint(rangeY[0], rangeY[1]),randint(rangez[0], rangez[1])) ######差z轴输入
        while self.map[0][z][y][x] == 1:
            x, y,z = (randint(rangeX[0], rangeX[1]), randint(rangeY[0], rangeY[1]),randint(rangez[0], rangez[1]))
        return (x, y,z)

    def showMap(self):
        print("+" * (3 * self.width + 2))

        for row in self.map[0]:
            s = '+'
            for entry in row:
                s += ' ' + str(entry) + ' '
            s += '+'
            print(s)

        print("+" * (3 * self.width + 2))


def AStarSearch(map, source, dest,volume_x,volume_y,volume_z):

    def getNewPosition(map, location, offset):

        x, y,z = (location.x + offset[0], location.y + offset[1],location.z + offset[2])
        t = location.t + int(1)
        int(t)
        if x < 0 or x >= map.width or y < 0 or y >= map.height or z < 0 or z >= map.Zlength: 
            return None
        
        if isObstacleNearby(map, x, y, z,t):# map.map[t][y][x] 
            return None

        return (x, y,z,t)
    def isObstacleNearby(map, x, y, z,t): #判断当勤粒子附近的障碍物
        int(t)

        x_min, x_max = max(x - volume_x, 0), min(x + volume_x, map.width - 1)
        y_min, y_max = max(y - volume_y, 0), min(y + volume_y, map.height - 1)
        z_min, z_max = max(z - volume_z, 0), min(z + volume_z, map.Zlength - 1)
        
        # 使用切片检查范围内是否有障碍物
        # map_array = np.array(map.map) #转换
        #print(map_array.shape)
        if np.any(map.map[t:t+1, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] == 1):
            return True
        return False
    def getPositions(map, location):
        offsets = [ (-1, 0,0),#x左
                    ( 1, 0,0),#x右
                    ( 0,-1,0),#y后
                    ( 0, 1,0),#y前
                    ( 0, 0,1),#z上
                    ( 0, 0,-1),#z下
                    (-1, 1,0),#xy左前
                    (-1,-1,0),#xy左后
                    ( 1, 1,0),#xy右前
                    ( 1,-1,0),#xy右后
                    ( 0,-1,1),#yz后上
                    ( 0,-1,-1),#yz后下                  
                    ( 0, 1,1),#yz前上
                    ( 0, 1,-1),#yz前下
                    ( 1, 0,-1),#xz右下
                    (-1, 0,-1),#xz左下
                    ( 1, 0, 1),#xz右上
                    (-1, 0, 1),#xz左上
                    ( 1, 1, 1),#xyz右前上
                    ( 1, 1, -1),#xyz右前下
                    ( 1,-1, -1),#xyz右后下
                    ( 1,-1, 1),#xyz右后上
                    ( -1, 1, 1),#xyz左前上
                    ( -1, -1, 1),#xyz左后上
                    ( -1, -1, -1),#xyz左后下
                    (-1, 1, -1),# xyz左前下    
                    (0,0,0)#等待

                    
                    ]
        poslist = []
        for offset in offsets:
            pos = getNewPosition(map, location, offset)
            if pos is not None:
                poslist.append(pos)
        return poslist

    def calHeuristic(pos, dest):
        return abs(dest.x - pos[0]) + abs(dest.y - pos[1])+ abs(dest.z - pos[2])#math.sqrt((dest.x - pos[0])**2 + (dest.y - pos[1])**2 + (dest.z - pos[2])**2)#abs(dest.x - pos[0]) + abs(dest.y - pos[1])+ abs(dest.z - pos[2])
        # pos_array = np.array(pos)
        # dest_array = np.array([dest.x, dest.y, dest.z])
        # # 计算欧几里得距离
        # distance = np.linalg.norm(dest_array - pos_array)
        # return distance
    def getMoveCost(location, pos):
        dx = abs(location.x - pos[0])
        dy = abs(location.y - pos[1])
        dz = abs(location.z - pos[2])

        # 根据移动的维度数量确定移动成本
        if dx + dy + dz == 1:
            return 1  # 沿一个轴移动
        elif dx + dy + dz == 2:
            return 1.4  # 沿两个轴移动（对角线）
        elif dx + dy + dz == 3:
            return 1.7  # 沿三个轴移动（立体对角线）
        elif dx + dy + dz == 0:
            return 1
        else:
            return 10  # 不合法的移动
    def isInList(list, pos):
        if pos in list:
            return list[pos]
        return None

    def addAdjacentPositions(map, location, dest, openlist, closedlist,source):
        poslist = getPositions(map, location)
        for pos in poslist:
            if isInList(closedlist, pos) is not None:
                continue

            findEntry = isInList(openlist, pos)
            # cross_product=inclination(source, dest,location)
            # cross_magnitude = np.linalg.norm(cross_product)
            h_cost = calHeuristic(pos, dest) #+ 0.01*cross_magnitude
            g_cost = location.g_cost + getMoveCost(location, pos)
            if findEntry is None:
                openlist[pos] = SearchEntry(pos[0], pos[1],pos[2],pos[3], g_cost, g_cost + h_cost, location)
            elif findEntry.g_cost > g_cost:
                findEntry.g_cost = g_cost
                findEntry.f_cost = g_cost + h_cost
                findEntry.pre_entry = location


    def inclination(source, dest,location):
        # 计算向量
        v1 = np.array([location.x - dest.x, location.y - dest.y, location.z - dest.z])
        v2 = np.array([source[0] - dest.x, source[1] - dest.y, source[2] - dest.z])
        return np.cross(v1, v2)
    
    def getFastPosition(openlist):
        fast = None
        for entry in openlist.values():
            if fast is None:
                fast = entry
            elif fast.f_cost > entry.f_cost:
                fast = entry
        return fast
    t=0
    volume_x=volume_x
    volume_y=volume_y
    volume_z=volume_z
    openlist, closedlist = {}, {}
    location = SearchEntry(source[0], source[1],source[2],t, 0.0)# source[2] == z coordiante of start
    dest = SearchEntry(dest[0], dest[1],dest[2],t, 0.0) #dest == z coordinate of goal
    openlist[(source[0], source[1], source[2], t)] = location
    #print(openlist)
  
    times=0
    while True:
        location = getFastPosition(openlist) # search possible avaliable position as start
        if location is None or times>=300: #300means time limit
            #print("can't find valid path!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(i)
            break

        if location.x == dest.x and location.y == dest.y and location.z == dest.z:#
            break
#         # 先检查 Key 是否存在
# # Debug 日志
#         print(f"Trying to pop {location.getPos()} from openlist.")
#         print(f"Current openlist keys: {list(openlist.keys())}")

#         # 先检查 Key 是否存在
#         if location.getPos() in openlist:
#             openlist.pop(location.getPos())
#         else:
#             print(f"Warning: {location.getPos()} not found in openlist!")
        closedlist[location.getPos()] = location
        openlist.pop(location.getPos())
        addAdjacentPositions(map, location, dest, openlist, closedlist,source)
        times=times+1
        
    path = []
    while location is not None:
        path.append((location.x, location.y,location.z))
       # map.map[location.y][location.x] = 2
        location = location.pre_entry
    path.reverse()
    #print(path)
    return path
import os
class PathManager:
    def __init__(self):
        # 初始化 DataFrame 以存储路径数据
        self.df = pd.DataFrame(columns=['Path_ID', 'X', 'Y', 'Z'])
        self.path_count = 0  # 跟踪路径数量

    def add_path(self, path):
        """添加一条新路径到 DataFrame 中"""
        temp_df = pd.DataFrame(path, columns=['X', 'Y', 'Z'])
        temp_df['Path_ID'] = self.path_count
        self.df = pd.concat([self.df, temp_df], ignore_index=True)
        self.path_count += 1  # 更新路径计数

    def save_paths_to_csv(self, file_name):
        """将所有路径保存到一个 CSV 文件中"""
        base_name, extension = os.path.splitext(file_name)
        counter = 1
        # 如果文件存在，找到一个新的文件名
        while os.path.exists(file_name):
            file_name = f"{base_name}{counter}{extension}"
            counter += 1

        # 保存 DataFrame 到 CSV 文件
        self.df.to_csv(file_name, index=False)
        print(f"Saved all paths to {file_name}")

    def get_paths(self):
        """返回当前所有路径的 DataFrame"""
        return self.df
import random
import time
def generate_unique_coordinates(num_particles, coord_range, x_min_dist, y_min_dist, z_min_dist):
    def is_valid(new_coord, existing_coords):
        for coord in existing_coords:
            if (abs(coord[0] - new_coord[0]) < x_min_dist or
                abs(coord[1] - new_coord[1]) < y_min_dist or
                abs(coord[2] - new_coord[2]) < z_min_dist):
                return False
        return True

    coordinates = []
    while len(coordinates) < num_particles:
        new_coord = (random.randint(0, coord_range), random.randint(0, coord_range), random.randint(0, coord_range))
        if not coordinates or is_valid(new_coord, coordinates):
            coordinates.append(new_coord)
    return coordinates
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def optimal_assignment(start_points, end_points):
    cost_matrix = np.zeros((len(start_points), len(end_points)))
    for i, start in enumerate(start_points):
        for j, end in enumerate(end_points):
            cost_matrix[i][j] = distance(start, end)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind
if __name__ == '__main__':
    WIDTH = 60
    HEIGHT = 60
    Zlength = 60
    i=0
    NUM_Particles=3
    Num_paths=10
    # set_of_source=[(1, 59, 30), (1, 24, 2), (13, 29, 8), (16, 0, 25), (25, 46, 52), (28, 44, 0), (36, 11, 15), (59, 26, 41)]
    # set_of_dest=[(15, 57, 29), (2, 16, 0), (7, 47, 3), (30, 20, 37), (31, 30, 53), (36, 56, 8), (44, 30, 12), (59, 23, 1)]
    coord_range = 59
    seed(89)
    volume_scale=0.5
    volume_x=int(4*volume_scale)
    volume_y=int(4*volume_scale)
    volume_z=int(8*volume_scale)
   
    x_min_dist = int(volume_x*2)
    y_min_dist = int(volume_y*2)
    z_min_dist = int(volume_z*2)
    Time=300
    start_time = time.time()
    for i in range(Num_paths):

        map = Map(WIDTH, HEIGHT,Zlength,Time,volume_x,volume_y,volume_z)
        set_of_source = generate_unique_coordinates(NUM_Particles, coord_range, x_min_dist, y_min_dist, z_min_dist)
        set_of_dest = generate_unique_coordinates(NUM_Particles, coord_range, x_min_dist, y_min_dist, z_min_dist)
        # 获得最优配对的终点顺序
        col_ind = optimal_assignment(set_of_source, set_of_dest)
        # 根据最优配对调整终点列表
        adjusted_end_points = [set_of_dest[i] for i in col_ind]

        paths=[]
    
        for o in range (NUM_Particles):

            #map.createBlock(BLOCK_NUM)
            
            source=set_of_source[o]
            dest=set_of_dest[o]#adjusted_end_points[o] zhege shi you tiaozhengde
            # print("source:", source)
            # print("dest:", dest)
            # print("Agent_number",i)
            path=AStarSearch(map, source, dest,volume_x,volume_y,volume_z)
            map.createBlock(path)
            #print(path)
            #print(len(path))
            paths.append(path)
        path_manager = PathManager()
        for path in paths:
            path_manager.add_path(path)

        # 保存路径到 CSV 文件
        #path_manager.save_paths_to_csv(f'{NUM_Particles}/{volume_scale}/all_paths.csv')     
    end_time = time.time()
    #  map.showMap()