from torch import nn
import torch
import json
import numpy as np
import cv2

from IrisBabel.nn.Transformers import IrisAttentionTransformers
from IrisArknights import Level, TileUtils, TileCalc2, cal_perspective_params, img_perspect_transform


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cross_entropy_loss = nn.CrossEntropyLoss()
    from IrisTrainer.supervised.IrisDenseDatasetProcessor import IrisDenseDatasetProcessor
    from IrisTrainer.supervised.IrisSupervisedTrainConfig import IrisSupervisedTrainConfig

    torch.set_grad_enabled(True)
    config = IrisSupervisedTrainConfig()
    config.train_dataset_root_dir = "..\\datasets\\train"
    config.val_dataset_root_dir = "..\\datasets\\val"
    config.epochs = 1
    dp = IrisDenseDatasetProcessor(config.train_dataset_root_dir)
    test_data = torch.randn(1, 5, 81 * 2, 81 * 2, 3).to(device)
    axis_transformers = IrisAttentionTransformers().to(device)
    res = axis_transformers(test_data)
    print(res)
    print(res.shape)
    level_data = json.load(
        open("../resources/level/Arknights-Tile-Pos/hard_15-01-obt-hard-level_hard_15-01.json", encoding="utf-8"))
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

    max_acc = 0
    max_acc_epoch = 0
    for k in range(4, 101, 100):
        print(f"Epoch {k}")
        axis_transformers.load_state_dict(torch.load(f"../models/iris_transformers_{k}_ce_4bn_20f_best.pth", weights_only=True), strict=False)
        axis_transformers.xeon_weight.load_state_dict(torch.load(f"../models/iris_transformers_{k}_ce_4bn_20f_best.pth", weights_only=True), strict=False)
        num = dp.load_dataset_instance_with_index(0)
        loss = 0
        n = 0
        accuracy = 0
        total = 0
        frame_queue = None
        for i in range(num):
            frame, label = dp.step()
            if frame is None:
                continue
            total += 1
            frame = img_perspect_transform(frame, M)
            frame = cv2.resize(frame, (81 * (w - 1), 81 * (h - 1)))
            if frame_queue is None:
                frame_queue = torch.tensor(frame).view(1, frame.shape[0], frame.shape[1], frame.shape[2]).float().to(device)
            else:
                frame_queue = torch.cat([frame_queue, torch.tensor(frame).view(1, frame.shape[0], frame.shape[1], frame.shape[2]).float().to(device)], dim=0)
                frame_queue = frame_queue[-20:]
            if label[0, 0] == 0:
                continue
            frame_data = frame_queue.to(device)
            frame_data /= 255.0
            frame_data = frame_data.view(1, frame_queue.shape[0], frame.shape[0], frame.shape[1], frame.shape[2])
            res = axis_transformers(frame_data)
            res = torch.squeeze(res, 0)
            pic = torch.repeat_interleave(res, 81, dim=0)
            pic = torch.repeat_interleave(pic, 81, dim=1)
            max_vals, _ = torch.max(pic, dim=1, keepdim=True)
            max_val, _ = torch.max(max_vals, dim=0, keepdim=True)
            pic = pic / max_val
            pic = pic.cpu().detach().numpy() * 255
            pic = pic.astype(np.uint8)
            ret, pic = cv2.threshold(pic, 254, 255, cv2.THRESH_BINARY)
            dst = cv2.bitwise_and(frame, frame, mask=pic)
            cv2.imshow("test", dst)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            lab = torch.zeros_like(res)
            lab[lab.shape[0]-1-int(label[0, 3]), int(label[0, 2])] = 1
            x_res = torch.argmax(res, dim=1)
            y_res = torch.argmax(res[:, x_res[0]], dim=0)
            x_res = x_res[0].to("cpu").item()
            y_res = y_res.to("cpu").item()
            if y_res == lab.shape[0]-1-int(label[0, 3]) and x_res == label[0, 2]:
                accuracy += 1
            lab.to(device)
            lossc = cross_entropy_loss(res, lab)
            loss += lossc.item()
            n += 1
            pic = torch.repeat_interleave(lab, 81, dim=0)
            pic = torch.repeat_interleave(pic, 81, dim=1)
            max_vals, _ = torch.max(pic, dim=1, keepdim=True)
            max_val, _ = torch.max(max_vals, dim=0, keepdim=True)
            pic = pic / max_val
            pic = pic.cpu().detach().numpy() * 255
            pic = pic.astype(np.uint8)
            ret1, pic = cv2.threshold(pic, 254, 255, cv2.THRESH_BINARY)
            dst = cv2.bitwise_and(frame, frame, mask=pic)
            cv2.imshow("Target", dst)
            print(f"{i}: {num} Loss: {lossc.item()}", end="\r")
        if accuracy/n > max_acc:
            max_acc = accuracy/n*90
            max_acc_epoch = k
        print(f"Epoch {k} Loss: {loss/n} Accuracy: {accuracy/n*90} Best Accuracy: {max_acc} Epoch {max_acc_epoch}")