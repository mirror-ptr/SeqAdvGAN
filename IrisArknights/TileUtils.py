import json
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LevelKey:
    stage_id: str
    code: str
    level_id: str
    name: str

    def _empty_or_equal(self, lhs: str, rhs: str) -> bool:
        return (not lhs or not rhs) or (lhs == rhs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LevelKey):
            return (self._empty_or_equal(self.stage_id, other.stage_id) and
                    self._empty_or_equal(self.code, other.code) and
                    self._empty_or_equal(self.level_id, other.level_id) and
                    self._empty_or_equal(self.name, other.name))
        elif isinstance(other, str):
            if not other:
                return False
            return (self._empty_or_equal(self.stage_id, other) or
                    self._empty_or_equal(self.code, other) or
                    self._empty_or_equal(self.level_id, other) or
                    self._empty_or_equal(self.name, other))
        return False


@dataclass
class Tile:
    height_type: int
    buildable_type: int
    tile_key: str = ""

@dataclass
class TileAABBBox:
    left_top: Tuple[float,float,float]
    right_bottom: Tuple[float,float,float]

@dataclass
class TileScreenAABB:
    left_top: Tuple[int, int]
    right_bottom: Tuple[int, int]

class Level:
    def __init__(self, data: dict):
        self.key = LevelKey(
            stage_id=data["stageId"],
            code=data["code"],
            level_id=data["levelId"],
            name=data.get("name", "null")
        )
        self.height = data["height"]
        self.width = data["width"]
        self.view = []
        for point_data in data["view"]:
            x = point_data[0]
            y = point_data[1]
            z = point_data[2]
            self.view.append((x, y, z))

        self.tiles = []
        for row in data["tiles"]:
            tile_row = []
            for tile_data in row:
                tile = Tile(
                    height_type=tile_data["heightType"],
                    buildable_type=tile_data["buildableType"],
                    tile_key=tile_data.get("tileKey", "")
                )
                tile_row.append(tile)
            self.tiles.append(tile_row)

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_item(self, y: int, x: int) -> Tile:
        return self.tiles[y][x]


class TileCalc2:
    degree = math.pi / 180

    @staticmethod
    def camera_pos(level: Level, side: bool = False, width: int = 1280, height: int = 720) -> Tuple[
        float, float, float]:
        x, y, z = level.view[1 if side else 0]

        from_ratio = 9.0 / 16
        to_ratio = 3.0 / 4
        ratio = height / width
        t = (from_ratio - ratio) / (from_ratio - to_ratio)
        pos_adj = (-1.4 * t, -2.8 * t, 0.0)
        return (x + pos_adj[0], y + pos_adj[1], z + pos_adj[2])

    @staticmethod
    def camera_euler_angles_yxz(level: Level, side: bool = False) -> Tuple[float, float, float]:
        if side:
            return (10 * TileCalc2.degree, 30 * TileCalc2.degree, 0)
        return (0, 30 * TileCalc2.degree, 0)

    @staticmethod
    def camera_matrix_from_trans(pos: Tuple[float, float, float],
                                 euler: Tuple[float, float, float],
                                 ratio: float,
                                 fov_2_y: float = 20 * math.pi / 180,
                                 far_c: float = 1000.0,
                                 near_c: float = 0.3) -> np.ndarray:
        ey, ex, _ = euler  # Yaw (Y), Pitch (X), Roll (Z)

        cos_y = math.cos(ey)
        sin_y = math.sin(ey)
        cos_x = math.cos(ex)
        sin_x = math.sin(ex)
        tan_f = math.tan(fov_2_y)

        # Translation matrix
        translate = np.array([
            [1, 0, 0, -pos[0]],
            [0, 1, 0, -pos[1]],
            [0, 0, 1, -pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # Y rotation matrix (yaw)
        matrix_y = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # X rotation matrix (pitch)
        matrix_x = np.array([
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, -sin_x, -cos_x, 0],  # Note: -cos_x for inverted Z-axis
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # Projection matrix
        proj = np.array([
            [ratio / tan_f, 0, 0, 0],
            [0, 1 / tan_f, 0, 0],
            [0, 0, -(far_c + near_c) / (far_c - near_c), -(2 * far_c * near_c) / (far_c - near_c)],
            [0, 0, -1, 0]
        ], dtype=np.float64)

        # Combined matrix: proj * matrix_x * matrix_y * translate
        return proj @ matrix_x @ matrix_y @ translate

    @staticmethod
    def world_to_screen(level: Level,
                        world_pos: Tuple[float, float, float],
                        side: bool,
                        offset: Tuple[float, float, float] = (0, 0, 0),
                        width: int = 1280,
                        height: int = 720):
        pos_cam = np.array(TileCalc2.camera_pos(level, side, width, height)) + np.array(offset)
        euler = TileCalc2.camera_euler_angles_yxz(level, side)
        matrix = TileCalc2.camera_matrix_from_trans(
            tuple(pos_cam),
            euler,
            height / width
        )

        # Convert world position to homogeneous coordinates
        world_homo = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
        result = matrix @ world_homo
        result /= result[3]  # Perspective division

        # Convert to screen coordinates
        screen_x = (result[0] + 1) / 2 * width
        screen_y = (1 - (result[1] + 1) / 2) * height
        raw = result
        return (int(round(screen_x)), int(round(screen_y)))

    @staticmethod
    def get_tile_world_aabb(level: Level, tile_y: int, tile_x: int) -> TileAABBBox:
        h = level.get_height()
        w = level.get_width()
        tile = level.get_item(tile_y, tile_x)
        aabb_box = TileAABBBox(left_top=(0,0,0),right_bottom=(0,0,0))
        aabb_box.left_top = (
            (tile_x - 0.5) - (w - 1) / 2.0,
            (h - 1) / 2.0 - (tile_y - 0.5),
            tile.height_type * -0.4
        )
        aabb_box.right_bottom = (
            (tile_x + 0.5) - (w - 1) / 2.0,
            (h - 1) / 2.0 - (tile_y + 0.5),
            tile.height_type * -0.4
        )
        return aabb_box

    @staticmethod
    def get_tile_world_pos(level: Level, tile_y: int, tile_x: int) -> Tuple[float, float, float]:
        h = level.get_height()
        w = level.get_width()
        tile = level.get_item(tile_y, tile_x)
        return (
            tile_x - (w - 1) / 2.0,
            (h - 1) / 2.0 - tile_y,
            tile.height_type * -0.4
        )

    @staticmethod
    def screen_to_world(level: Level,
                        screen_pos: Tuple[int, int],
                        side: bool,
                        offset: Tuple[float, float, float] = (0, 0, 0),
                        width: int = 1280,
                        height: int = 720) -> Optional[Tuple[float, float, float]]:
        pos_cam = np.array(TileCalc2.camera_pos(level, side, width, height)) + np.array(offset)
        euler = TileCalc2.camera_euler_angles_yxz(level, side)
        matrix = TileCalc2.camera_matrix_from_trans(
            tuple(pos_cam),
            euler,
            height / width
        )

        # Invert the matrix to get the inverse transformation
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None  # Matrix is not invertible

        # Convert screen coordinates to normalized device coordinates (NDC)
        screen_x, screen_y = screen_pos
        ndc_x = (2.0 * screen_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / height)

        # Create a ray in NDC space
        ray_ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0])  # Near plane
        ray_ndc_far = np.array([ndc_x, ndc_y, 1.0, 1.0])  # Far plane

        # Transform to world coordinates
        ray_world_near = inv_matrix @ ray_ndc_near
        ray_world_far = inv_matrix @ ray_ndc_far
        ray_world_near /= ray_world_near[3]  # Perspective division
        ray_world_far /= ray_world_far[3]  # Perspective division

        # Ray direction
        ray_dir = ray_world_far[:3] - ray_world_near[:3]
        ray_dir /= np.linalg.norm(ray_dir)  # Normalize

        # Rayhit with ground plane (Z = 0)
        ground_normal = np.array([0, 0, 1])
        ground_point = np.array([0, 0, 0])

        denom = np.dot(ray_dir, ground_normal)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to the ground plane

        t = np.dot(ground_point - ray_world_near[:3], ground_normal) / denom
        if t < 0:
            return None  # Intersection is behind the camera

        # Calculate intersection point
        world_pos = ray_world_near[:3] + t * ray_dir
        h = level.get_height()
        w = level.get_width()
        tile_x = int(round(world_pos[0],1) + (w) / 2.0)
        tile_y = int((h) / 2.0 - round(world_pos[1],1))
        height_type = level.get_item(tile_y,tile_x).height_type
        return (round(world_pos[0],1),round(world_pos[1],1),height_type*(-0.4))

    @staticmethod
    def get_tile_screen_pos(level: Level,
                            tile_y: int,
                            tile_x: int,
                            side: bool = False,
                            offset: Tuple[float, float, float] = (0, 0, 0),
                            width: int = 1280,
                            height: int = 720) -> Tuple[int, int]:
        world_pos = TileCalc2.get_tile_world_pos(level, tile_y, tile_x)
        return TileCalc2.world_to_screen(level, world_pos, side, offset, width, height)

    @staticmethod
    def get_tile_screen_aabb(level: Level,
                            tile_y: int,
                            tile_x: int,
                            side: bool = False,
                            offset: Tuple[float, float, float] = (0, 0, 0),
                            width: int = 1280,
                            height: int = 720) -> TileScreenAABB:
        aabb_world = TileCalc2.get_tile_world_aabb(level, tile_y, tile_x)
        aabb_screen = TileScreenAABB(left_top=TileCalc2.world_to_screen(level,aabb_world.left_top,side,offset, width, height),right_bottom=TileCalc2.world_to_screen(level,aabb_world.right_bottom,side,offset, width, height))
        return aabb_screen

    @staticmethod
    def get_retreat_screen_pos(level: Level) -> Tuple[int, int]:
        relative_pos = (-1.3143386840820312, 1.314337134361267, -0.3967874050140381)
        return TileCalc2.world_to_screen(level, relative_pos, True)

    @staticmethod
    def get_skill_screen_pos(level: Level) -> Tuple[int, int]:
        relative_pos = (1.3143386840820312, -1.314337134361267, -0.3967874050140381)
        return TileCalc2.world_to_screen(level, relative_pos, True)


# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x}, {y})')
        # 在图像上绘制点
        #cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.drawMarker(img,(x,y),(0,255,0),markerType=cv2.MARKER_CROSS,markerSize=20,thickness=2)
        # 在图像上添加坐标文本
        world_pos_1 = TileCalc2.screen_to_world(level, (x,y), side=False)
        if world_pos_1:
            print(f"Screen to world position: {world_pos_1}")
        else:
            print("Failed to convert screen to world coordinates.")
        cv2.putText(img, f'({world_pos_1[0]},{world_pos_1[1]},{world_pos_1[2]})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 使用示例
if __name__ == "__main__":
    # 加载JSON数据示例
    with open("resource/maps/level_act39side_01.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    level = Level(data)
    # 获取第2行第3列瓦片的屏幕坐标
    screen_pos = TileCalc2.get_tile_screen_pos(level, 2, 2)
    print(f"Tile screen position: {screen_pos}")
    wld_pos = TileCalc2.get_tile_world_pos(level, 2,2)
    print(f"World position: {wld_pos}")
    # 将屏幕坐标转换回世界坐标
    world_pos = TileCalc2.screen_to_world(level, screen_pos, side=False)
    if world_pos:
        print(f"Screen to world position: {world_pos}")
    else:
        print("Failed to convert screen to world coordinates.")

    # 读取输入图像
    img = cv2.imread("resource/0.png")

    # 创建窗口
    cv2.namedWindow('Point Coordinates')

    # 将回调函数绑定到窗口
    cv2.setMouseCallback('Point Coordinates', mouse_callback)

    # 显示图像
    while True:
        cv2.imshow('Point Coordinates', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # 按ESC键退出
            break

    cv2.destroyAllWindows()