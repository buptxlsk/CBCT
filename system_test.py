import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from numpy.testing.print_coercion_tables import print_new_cast_table


class TEST:
    def __init__(self):
        self.euler_angles_map = [0]*3
        self.center = np.array([383.5, 383.5, 287.5 ])
        self.origin_world = np.array([0,0,0])
        self.origin_physical = np.array([0,0,0])
        self.origin_physical_map = np.array([0,0,0])
        self.slice_thickness = 0.3
        self.euler_angles = [0]*3
        self.PT = []
        self.vector_axial = ()
        self.vector_coronal = ()
        self.vector_sagittal = ()
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.s = None




    def euler_angles_from_rotation_matrix(self,matrix):
        """从旋转矩阵计算欧拉角"""
        # 计算y角
        y = np.arctan2(matrix[0, 2], np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))

        # 检查是否接近万向锁情况
        if np.abs(y - np.pi / 2) < 1e-6:
            # 万向锁情况，y = 90度
            print("警告：检测到万向锁情况（y = 90度）")
            z = 0
            x = np.arctan2(matrix[1, 0], matrix[1, 1])
        elif np.abs(y + np.pi / 2) < 1e-6:
            # 万向锁情况，y = -90度
            print("警告：检测到万向锁情况（y = -90度）")
            z = 0
            x = np.arctan2(-matrix[1, 0], -matrix[1, 1])
        else:
            # 一般情况
            z = np.arctan2(-matrix[0, 1], matrix[0, 0])
            x = np.arctan2(-matrix[1, 2], matrix[2, 2])

        # 将弧度转换为角度
        x = np.degrees(x) % 360
        y = np.degrees(y) % 360
        z = np.degrees(z) % 360

        return x, y, z

    def normalize_vector(self, vector):
        """归一化向量"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def rotation_matrix_from_vectors(self, v1, v2, v3):
        """从三个向量构建旋转矩阵"""
        rotation_matrix = np.column_stack((v1, v2, v3))
        return rotation_matrix

    def rotate_coordinate(self, x, y, z, angle_x, angle_y, angle_z, reverse_turn=False):
        """旋转坐标点"""
        # 将角度从度转换为弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

        # 绕X、Y、Z轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z
        R = R.transpose()

        # 原始点向量
        original_vector = np.array([x, y, z])

        # 将点平移到原点
        translated_point = original_vector - self.center

        # 旋转点
        rotated_point = R @ translated_point

        # 将点平移回中心
        new_point = rotated_point + self.center

        return new_point

    def rotate_coordinate_plus(self, x, y, z, angle_x, angle_y, angle_z, reverse_turn=False):
        """旋转坐标点"""
        # 将角度从度转换为弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

        # 绕X、Y、Z轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z
        R = R.transpose()

        # 原始点向量
        original_vector = np.array([x, y, z])

        # 将点平移到原点
        translated_point = original_vector - self.origin_world

        # 旋转点
        rotated_point = R @ translated_point

        # 将点平移回中心
        new_point = rotated_point + self.origin_world

        return new_point

    def update_physical_position_label_plus(self, x,y,z, angle_x=0, angle_y=0, angle_z=0,display = True):
        point = self.rotate_coordinate(x,y,z, -angle_x, -angle_y, -angle_z)
        print(point)

        pos = self.rotate_coordinate(point[0],point[1],point[2],
                                     self.euler_angles_map[0],
                                     self.euler_angles_map[1],
                                     self.euler_angles_map[2])
        print(f'以center为中心的坐标,图片欧拉角{pos}')


        pos = self.rotate_coordinate(point[0],point[1],point[2],
                                     self.euler_angles[0],
                                     self.euler_angles[1],
                                     self.euler_angles[2])
        print(f'以center为中心的坐标,欧拉角{pos}')

        slice_pos = np.array([pos[0],pos[1],pos[2]])-self.origin_physical
        position = [x * self.slice_thickness for x in slice_pos ]
        pos_plus = (position[0],position[1], position[2])
        print(f'新方法算出的物理坐标为{pos_plus}')

        return pos_plus

    def update_physical_position_label(self, x, y, z):
        slice_pos = np.array([x,y,z])-self.origin_physical
        position = [x * self.slice_thickness for x in slice_pos ]
        pos = (position[0],position[1], position[2])
        return pos

    def calculate_position_in_key_coordinates(self, x,y,z, angle_x, angle_y, angle_z):
        point = self.rotate_coordinate(x,y,z, -angle_x, -angle_y, -angle_z, True)
        pos = self.rotate_coordinate(point[0],point[1],point[2],
                                self.euler_angles[0],
                                self.euler_angles[1],
                                self.euler_angles[2])
        return pos

    def calculate_position_in_key_coordinates_map(self, x,y,z, angle_x, angle_y, angle_z):
        point = self.rotate_coordinate(x,y,z, -angle_x, -angle_y, -angle_z, True)
        pos = self.rotate_coordinate(point[0],point[1],point[2],
                                self.euler_angles_map[0],
                                self.euler_angles_map[1],
                                self.euler_angles_map[2])
        return pos


    def set_coordinate_system(self):
        """设置坐标系"""
        key_points = [
            ('a', (389, 371, 225), (0, 0, 0), (0, 0, 0)),
            ('b', (380, 649, 199), (0, 0, 0), (0, 0, 0)),
            ('c', (281, 359, 303), (0, 0, 0), (0, 0, 0)),
            ('d', (509, 366, 303), (0, 0, 0), (0, 0, 0)),
            ('s', (387, 440, 269), (0, 0, 0), (0, 0, 0)),
        ]
        # key_points = [
        #     ('a', (358, 346, 169), (0, 0, 0), (0, 0, 0)),
        #     ('b', (370, 609, 216), (0, 0, 0), (0, 0, 0)),
        #     ('c', (240, 345, 277), (0, 0, 0), (0, 0, 0)),
        #     ('d', (469, 327, 277), (0, 0, 0), (0, 0, 0)),
        #     ('s', (360, 417, 261), (0, 0, 0), (0, 0, 0)),
        # ]


        # 此处作用存疑
        points = [self.rotate_coordinate(x, y, z, angle_x, angle_y, angle_z, True)
                  for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points]


        # 保留旋转前s点的世界坐标
        self.origin_world = points[4]

        # 计算向量AB和CD
        vector_AB = points[1] - points[0]
        vector_CD = points[3] - points[2]

        # print(vector_AB, vector_CD)

        # 计算水平面、冠状面和矢状面的法向量
        self.vector_axial = self.normalize_vector(np.cross(vector_AB, vector_CD))  # 水平面的法向量,新z轴
        self.vector_coronal = self.normalize_vector(np.cross(vector_CD,self.vector_axial))  # 冠状面的法向量，新y轴
        self.vector_sagittal = self.normalize_vector(np.cross(self.vector_coronal,self.vector_axial))  # 矢状面的法向量，新x轴

        # print(self.vector_sagittal)
        # print(self.vector_coronal)
        # print(self.vector_axial)

        # 构建旋转矩阵
        rotation_matrix = self.rotation_matrix_from_vectors(self.vector_sagittal, self.vector_coronal, self.vector_axial)
        print(f"rotation_matrix\n{rotation_matrix}")

        # 计算欧拉角
        self.euler_angles = self.euler_angles_from_rotation_matrix(rotation_matrix)

        print(f"euler_angles\n{self.euler_angles}")

        self.euler_angles_map[0] = self.euler_angles[0]
        self.euler_angles_map[1] = self.euler_angles[1]
        self.euler_angles_map[2] = self.euler_angles[2]

        self.euler_angles_map[0] = (180 + self.euler_angles[0]) % 360
        self.euler_angles_map[1] = -(self.euler_angles[1]) % 360
        self.euler_angles_map[2] = -(self.euler_angles[2]) % 360
        self.euler_angles_map[2] = 180 - self.euler_angles[2]


        print(f'euler_angles_map\n{self.euler_angles_map}')

        # 计算旋转后s点的世界坐标
        self.origin_physical = self.calculate_position_in_key_coordinates(self.origin_world[0], self.origin_world[1],self.origin_world[2],0,0,0)
        print(f"origin_physical\n{self.origin_physical}")

        self.origin_physical_map = self.calculate_position_in_key_coordinates_map(self.origin_world[0], self.origin_world[1],self.origin_world[2],0,0,0)
        print(f"origin_physical_map\n{self.origin_physical_map}")

        # 以转前s点为原点，保留转后的相对关系，求各个点的世界坐标，同时将相对关系存为物理坐标
        for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points:
            # 这个是初始世界坐标
            slice_pos = (x, y, z)
            # 这个是初始角度
            angles = (angle_x, angle_y, angle_z)
            physicals = self.update_physical_position_label_plus(x, y, z, angle_x, angle_y, angle_z)
            pt = (name, slice_pos, angles, physicals)
            if name == "a":
                self.a = pt
            elif name == "b":
                self.b = pt
            elif name == "c":
                self.c = pt
            elif name == "d":
                self.d = pt
            elif name == "s":
                self.s = pt
            self.PT.append(pt)

        pprint(self.PT)







def main():
    """主函数"""
    # 创建 TEST 类的实例
    test_instance = TEST()
    test_instance.set_coordinate_system()

if __name__ == "__main__":
    main()
