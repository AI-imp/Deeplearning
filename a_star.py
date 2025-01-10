"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import cv2
import matplotlib.pyplot as plt

show_animation = True
class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
    class Node:#节点
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost#代价
            self.parent_index = parent_index
        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):#网格作为最小单位将像素重索引
        #初始化节点
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        #初始化开集闭集
        open_set, closed_set = dict(), dict()
        #初始化开集的起始节点：起始节点
        open_set[self.calc_grid_index(start_node)] = start_node
        while True:
            if len(open_set) == 0:
                break
            c_id = min(open_set,key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,open_set[o]))
            current = open_set[c_id]
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break
            #删除当前节点
            del open_set[c_id]
            #添加到闭集
            closed_set[c_id] = current
            #选择运动模式
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],current.y + self.motion[i][1],current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)
                if not self.verify_node(node):#是障碍物
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if node.cost<open_set[n_id].cost:
                        open_set[n_id] = node
        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False
        # collision check
        if self.obstacle_map[node.x][node.y]:#检查当前节点是否在障碍物中
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)#障碍物x域
        self.y_width = round((self.max_y - self.min_y) / self.resolution)#障碍物y域
        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]#以false初始化障碍物列表
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)#格子的实际x坐标
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)#欧式距离
                    if d <= self.rr:#如果<=表示障碍物与车体相交
                        self.obstacle_map[ix][iy] = True#将对应格子设为true表示有障碍物
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion


def main():
    print(__file__ + " start!!")
    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 100.0  # [m]
    gy = 100.0  # [m]
    grid_size = 100.0  # [m]
    robot_radius = 1.0  # [m]
    ox, oy = [], []#关闭列表/石头
    img = cv2.imread(r'E:\cource\u3exercise\img.png', cv2.IMREAD_GRAYSCALE)
    rec_a = img.shape[0]
    rec_h = img.shape[1]
    for i in range(0,rec_a):
        for j in range(0, rec_h):
            if img[i][j]!=255:
                ox.append(i)
                oy.append(j)
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()
if __name__ == '__main__':
    main()

