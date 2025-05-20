import json
import cv2
import numpy as np
from IrisArknights import TileCalc2, Level, cal_perspective_params, img_perspect_transform, draw_line

if __name__ == '__main__':
    level_data = json.load(open("../resources/level/Arknights-Tile-Pos/hard_15-01-obt-hard-level_hard_15-01.json", encoding="utf-8"))
    level = Level(level_data)
    w = level.get_width()
    h = level.get_height()
    left_top = ((0 - 0.5) - (w - 1) / 2.0,
                (h - 1) / 2.0 - (0 - 0.5),
                0)
    right_top = ((w - 2 + 0.5) - (w - 1) / 2.0,
                 (h - 1) / 2.0 - (0 - 0.5),
                 0)
    left_bottom = ((0 - 0.5) - (w - 1) / 2.0,
                   (h - 1) / 2.0 - (h - 2 + 0.5),
                   0)
    right_bottom = ((w - 2 + 0.5) - (w - 1) / 2.0,
                    (h - 1) / 2.0 - (h - 2 + 0.5),
                    0)
    left_top = TileCalc2.world_to_screen(level, left_top, False, width=1920, height=1080)
    right_top = TileCalc2.world_to_screen(level, right_top, False, width=1920, height=1080)
    left_bottom = TileCalc2.world_to_screen(level, left_bottom, False, width=1920, height=1080)
    right_bottom = TileCalc2.world_to_screen(level, right_bottom, False, width=1920, height=1080)


    M, M_Inverse = cal_perspective_params([1920, 1080], np.array([list(left_top), list(right_top), list(left_bottom), list(right_bottom)]))
    cap = cv2.VideoCapture("../resources/video/H15-1.mp4")
    fourcc = cv2.VideoWriter.fourcc(*'MP4V')  # 视频编解码器
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    out = cv2.VideoWriter('../resources/result.mp4', fourcc, fps, (81*(w-1), 81*(h-1)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_line(frame, left_top, left_bottom, right_bottom, right_top)
        transformed_frame = img_perspect_transform(frame, M)
        transformed_frame = cv2.resize(transformed_frame, (81*(w-1), 81*(h-1)))
        out.write(transformed_frame)
        cv2.imshow('frame', transformed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def calculate_transform_matrix(json_file):
    level_data = json.load(
        open(json_file, encoding="utf-8"))
    level = Level(level_data)
    w = level.get_width()
    h = level.get_height()
    left_top = ((0 - 0.5) - (w - 1) / 2.0,
                (h - 1) / 2.0 - (0 - 0.5),
                0)
    right_top = ((w - 2 + 0.5) - (w - 1) / 2.0,
                 (h - 1) / 2.0 - (0 - 0.5),
                 0)
    left_bottom = ((0 - 0.5) - (w - 1) / 2.0,
                   (h - 1) / 2.0 - (h - 2 + 0.5),
                   0)
    right_bottom = ((w - 2 + 0.5) - (w - 1) / 2.0,
                    (h - 1) / 2.0 - (h - 2 + 0.5),
                    0)
    left_top = TileCalc2.world_to_screen(level, left_top, False, width=1920, height=1080)
    right_top = TileCalc2.world_to_screen(level, right_top, False, width=1920, height=1080)
    left_bottom = TileCalc2.world_to_screen(level, left_bottom, False, width=1920, height=1080)
    right_bottom = TileCalc2.world_to_screen(level, right_bottom, False, width=1920, height=1080)

    M, M_Inverse = cal_perspective_params([1920, 1080], np.array(
        [list(left_top), list(right_top), list(left_bottom), list(right_bottom)]))
    return level, M, M_Inverse

def transform_image(_level, _M, image):
    transformed_image = img_perspect_transform(image, _M)
    transformed_image = cv2.resize(transformed_image, (81 * (_level.get_width() - 1), 81 * (_level.get_height() - 1)))
    return transformed_image