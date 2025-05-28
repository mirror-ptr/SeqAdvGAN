import torch
from torch import nn
from IrisBabel.nn.Transformers import IrisMapEncoder, IrisBabelTransformer, IrisRTDetrVisionEncoder, IrisAttentionTransformers
from IrisBabel.nn.CNN import IrisXeonNet, IrisTriggerNet
import cv2
from IrisBabel.IrisBabelModel import IrisBabelModel
from IrisArknights import TileCalc2, Level, cal_perspective_params, img_perspect_transform, draw_line
import json
import numpy as np
from transformers import optimization

if __name__ == "__main__":
    print("Testing IrisBabel")
    print(f"Torch version: {torch.__version__}")
    print("Testing CUDA")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Device: CUDA")
    else:
        print("Device: CPU")
        exit()
    #x = torch.randn(1, 12, 10, 10, 1024)
    #test_map_encoder = IrisMapEncoder()
    #result = test_map_encoder(x)
    #print("===================Encoder Output===================")
    #print(result)
    #print(f"Shape: {result.shape}")
#
    test_img = cv2.imread("test_img/test.jpg", cv2.IMREAD_COLOR_RGB)
    #test_rtdetr_encoder = IrisRTDetrVisionEncoder()
    #result = test_rtdetr_encoder(test_img)
    #print(result)
    #print(f"Shape: {result.shape}")
#
    #action_list = []
    action_map = ["Idle", "Deploy", "Withdraw", "ActivateSkill", "DeactivateSkill"]
    #test_tgt_seq = torch.zeros(1, 1, 4096)
    #test_src_seq = torch.zeros(1, 1, 4096)
    #vocab_embedding = torch.randn(4,4096)
#
    ## A Valid Action must be constructed with the format below
    ## ActionName Target RelativePosition Direction
    ## which according to the output sequence:
    ## [-1]       [-2]   [-3]             [-4]
#
    #test_babel_transformer = IrisBabelTransformer()
    #for j in range(1):
    #    # map encoder
    #    #x = torch.randn(1, 12, 10, 10, 1024)
    #    #result = test_map_encoder(x)
#
    #    # end2end vit
    #    result = test_rtdetr_encoder(test_img)
    #    result = result.view(1, 1, 4096)
    #    test_src_seq = torch.cat([test_tgt_seq, result], dim=1)
    #    result = test_babel_transformer(test_src_seq, test_tgt_seq)
    #    temperature = 0.2
    #    next_action_probs = result.squeeze()[-1].squeeze() / temperature
    #    # Only Support 1 batch here
    #    next_action_probs = nn.functional.softmax(next_action_probs, dim=-1)
    #    next_action_id = torch.argmax(next_action_probs).item()
    #    print("===================Decoder Output====================")
    #    print(result)
    #    print(f"Shape: {result.shape}")
    #    print(f"Next action: {next_action_id}")
    #    action_list.append(next_action_id)
    #    action_embedding = vocab_embedding[next_action_id]
    #    action_embedding = action_embedding.view(1, 1, 4096)
    #    test_tgt_seq = torch.cat([test_tgt_seq, action_embedding], dim=1)
    #for action_id in action_list:
    #    print(f"Action: {action_map[action_id]}")

    #print("=======================Model Test========================")
    #model = IrisBabelModel()
    #for i in range(110):
    #    print(f"========================Prediction {i}=============================")
    #    action_id, action_target, action_pos, action_direction = model.pred(test_img)
    #    print(f"Action: {action_map[action_id]}")
    #    print(f"Action target: {action_target}")
    #    print(f"Action pos: {action_pos}")
    #    print(f"Action direction: {action_direction}")
    #print(f"Action Memory: {model.action_memory}, Shape: {model.action_memory.shape}")
    #print(f"Feature Memory: {model.feature_memory}, Shape: {model.feature_memory.shape}")
    #model = IrisBabelModel()
    #print("=======================Model Training Test========================")
    #model.summary()
    #model.train_backbone_start()
    #label = torch.tensor([[1, 4, 0.5, 0.5, 3]]).to(model.device)
    #model.lock_memory()
    #for i in range(110):
    #    print(f"======================Epoch {i}============================")
    #    action_id, action_target, action_pos, action_direction, action_id_probs, action_target_probs, action_direction_probs = model.train_backbone_step(test_img)
    #    print(f"Action: {action_map[action_id]}")
    #    print(f"Action target: {action_target}")
    #    print(f"Action pos: {action_pos}")
    #    print(f"Action direction: {action_direction}")
    #    loss = model.calculate_loss_and_backward(action_id, action_id_probs, action_target_probs, action_pos, action_direction_probs, label)
    #    print(f"Loss: {loss}")
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    from IrisTrainer.supervised.IrisDenseDatasetProcessor import IrisDenseDatasetProcessor
    from IrisTrainer.supervised.IrisSupervisedTrainConfig import IrisSupervisedTrainConfig
    torch.set_grad_enabled(True)
    config = IrisSupervisedTrainConfig()
    config.train_dataset_root_dir = "..\\datasets\\train"
    config.val_dataset_root_dir = "..\\datasets\\val"
    config.epochs = 1
    dp = IrisDenseDatasetProcessor(config.train_dataset_root_dir)
    test_data = torch.randn(1, 5, 81*2, 81*2, 3).to(device)
    axis_transformers = IrisAttentionTransformers().to(device)
    trigger_net = IrisTriggerNet().to(device)
    res = axis_transformers(test_data)
    weight_matrix = res.clone().detach()
    res1 = trigger_net(test_data, weight_matrix)
    print(res)
    print(res.shape)
    print(res1)
    print(res1.shape)
    level_data = json.load(open("../resources/level/Arknights-Tile-Pos/hard_15-01-obt-hard-level_hard_15-01.json", encoding="utf-8"))
    level = Level(level_data)
    lw = level.get_width()
    lh = level.get_height()
    left_top = ((0 - 0.5) - (lw - 1) / 2.0,
                (lh - 1) / 2.0 - (0 - 0.5),
                0)
    right_top = ((lw - 2 + 0.5) - (lw - 1) / 2.0,
                 (lh - 1) / 2.0 - (0 - 0.5),
                 0)
    left_bottom = ((0 - 0.5) - (lw - 1) / 2.0,
                   (lh - 1) / 2.0 - (lh - 2 + 0.5),
                   0)
    right_bottom = ((lw - 2 + 0.5) - (lw - 1) / 2.0,
                    (lh - 1) / 2.0 - (lh - 2 + 0.5),
                    0)
    left_top = TileCalc2.world_to_screen(level, left_top, False, width=1920, height=1080)
    right_top = TileCalc2.world_to_screen(level, right_top, False, width=1920, height=1080)
    left_bottom = TileCalc2.world_to_screen(level, left_bottom, False, width=1920, height=1080)
    right_bottom = TileCalc2.world_to_screen(level, right_bottom, False, width=1920, height=1080)


    M, M_Inverse = cal_perspective_params([1920, 1080], np.array([list(left_top), list(right_top), list(left_bottom), list(right_bottom)]))
   # cap = cv2.VideoCapture("../resources/video/H15-1.mp4")
   # while cap.isOpened():
   #     ret, frame = cap.read()
   #     if not ret:
   #         break
   #     transformed_frame = img_perspect_transform(frame, M)
   #     transformed_frame = cv2.resize(transformed_frame, (81 * (w - 1), 81 * (h - 1)))
#
   #     frame_data = torch.tensor(transformed_frame).float().to(device)
   #     frame_data = frame_data.view(1, 1, frame_data.shape[0], frame_data.shape[1], frame_data.shape[2])
   #     res = axis_transformers(frame_data)
   #     print(res)
   #     print(res.shape)
   #     res = torch.squeeze(res, 0)
   #     pic = torch.repeat_interleave(res, 81, dim=0)
   #     pic = torch.repeat_interleave(pic, 81, dim=1)
   #     max_vals, _ = torch.max(pic, dim=1, keepdim=True)
   #     max_val, _ = torch.max(max_vals, dim=0, keepdim=True)
   #     pic = pic / max_val
   #     pic = pic.cpu().detach().numpy()*255
   #     pic = pic.astype(np.uint8)
   #     ret, pic = cv2.threshold(pic, 254, 255, cv2.THRESH_BINARY)
#
   #     img_red = np.zeros([pic.shape[0], pic.shape[1], 3], np.uint8)
   #     img_red[:, :, 2] = np.zeros([pic.shape[0], pic.shape[1]]) + 255
   #     dst = cv2.bitwise_and(transformed_frame, transformed_frame, mask=pic)
   #     cv2.imshow("test", dst)
   #     if cv2.waitKey(1) & 0xFF == ord('q'):
   #         break
   # cap.release()
    trigger_net.load_state_dict(torch.load("../models/iris_trigger_ac_fl_0_ce_4bn_15f.pth", weights_only=True), strict=False)
    axis_transformers.load_state_dict(torch.load("../models/iris_transformers_ac_5_ce_4bn_20f_best.pth", weights_only=True), strict=False)
    #axis_transformers.xeon_weight.load_state_dict(torch.load("../models/iris_transformers_4_ce_4bn_20f_best.pth", weights_only=True), strict=False)
    optimizer = torch.optim.AdamW(axis_transformers.parameters(), lr=0.00001)
    optimizer1 = torch.optim.AdamW(trigger_net.parameters(), lr=0.0001)
    scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=4, num_training_steps=100, lr_end = 1e-7, power=3)
    #scheduler1 = optimization.get_polynomial_decay_schedule_with_warmup(optimizer1, num_warmup_steps=4, num_training_steps=100, lr_end = 1e-7, power=3)
    max_acc = 0
    max_acc_epoch = 0
    max_acc_tr = 0
    max_acc_epoch_tr = 0
    top_k = 5
    for k in range(1, 101):
        print(f"Epoch {k}")
        num = dp.load_dataset_instance_with_index(0)
        loss = 0
        loss_tr = 0
        n = 0
        accuracy = 0
        accuracy_tr = 0
        total = 0
        frame_queue = None
        for i in range(num):
            frame, label = dp.step()
            if frame is None:
                continue
            total += 1
            frame = img_perspect_transform(frame, M)
            frame = cv2.resize(frame, (81 * (lw - 1), 81 * (lh - 1)))
            if frame_queue is None:
                frame_queue = torch.tensor(frame).view(1, frame.shape[0], frame.shape[1], frame.shape[2]).float().to(device)
            else:
                frame_queue = torch.cat([frame_queue, torch.tensor(frame).view(1, frame.shape[0], frame.shape[1], frame.shape[2]).float().to(device)], dim=0)
                frame_queue = frame_queue[-15:]
            #if label[0, 0] == 0:
            #    continue
            frame_data = frame_queue.to(device)
            frame_data /= 255.0
            frame_data = frame_data.view(1, frame_queue.shape[0], frame_queue.shape[1], frame_queue.shape[2], frame_queue.shape[3])
            #frame_data1 = frame_data.clone()
            res = axis_transformers(frame_data)
            weight_matrix = res.clone().detach()
            res1 = trigger_net(frame_data, weight_matrix)
            res = torch.squeeze(res, 0)
            res1 = torch.squeeze(res1, 0)
            pic = torch.repeat_interleave(res.detach(), 81, dim=0)
            pic = torch.repeat_interleave(pic, 81, dim=1)
            max_vals, _ = torch.max(pic, dim=1, keepdim=True)
            max_val, _ = torch.max(max_vals, dim=0, keepdim=True)
            pic = pic / max_val
            pic = pic.cpu().detach().numpy() * 255
            pic = pic.astype(np.uint8)
            ret, pic = cv2.threshold(pic, 240, 255, cv2.THRESH_BINARY)
            dst = cv2.bitwise_and(frame, frame, mask=pic)
            #cv2.imshow("test", dst)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            #axis_transformers.zero_grad()
            trigger_net.zero_grad()
            lab = torch.zeros_like(res)
            lab[lab.shape[0]-1-int(label[0, 3]), int(label[0, 2])] = 1

            w, h = res.shape
            lres1 = res1.clone().detach().view(1, w*h, res1.shape[2])
            ltopk = torch.topk(res.view(1, w*h), top_k).indices
            zero8 = torch.zeros(8)
            zero8[0] = 1
            for j in ltopk[0]:
                lres1[0][j] = zero8
            if label[0, 0] != 0:
                lres1 = lres1.view(w, h, 8)
                zero8[0] = 0
                zero8[int(label[0, 0].item())] = 1
                lres1[lab.shape[0] - 1 - int(label[0, 3]), int(label[0, 2])] = zero8
                tmp1 = res1[lab.shape[0] - 1 - int(label[0, 3]), int(label[0, 2])].view(1, -1)
                _, max_i = torch.max(tmp1, dim=1)
                if max_i == int(label[0, 0].item()):
                    accuracy_tr += 1
            else:
                tmp1 = lres1[0][ltopk[0][0]].view(1, -1)
                _, max_i = torch.max(tmp1, dim=1)
                if max_i == 0:
                    accuracy_tr += 1
                lres1 = lres1.view(w, h, 8)
            lossc1 = cross_entropy_loss(lres1, res1)
            loss_tr += lossc1.item()
            lossc1.backward()
            optimizer1.step()
            #axis_transformers.zero_grad()

            x_res = torch.argmax(res, dim=1)
            y_res = torch.argmax(res[:, x_res[0]], dim=0)
            x_res = x_res[0].to("cpu").item()
            y_res = y_res.to("cpu").item()
            if y_res == lab.shape[0]-1-int(label[0, 3]) and x_res == label[0, 2]:
                accuracy += 1
            lab.to(device)
            lossc = cross_entropy_loss(res, lab)
            lossc = lossc
            #lossc.backward()
            loss += lossc.item()
            n += 1
            #optimizer.step()
            pic = torch.repeat_interleave(lab, 81, dim=0)
            pic = torch.repeat_interleave(pic, 81, dim=1)
            max_vals, _ = torch.max(pic, dim=1, keepdim=True)
            max_val, _ = torch.max(max_vals, dim=0, keepdim=True)
            pic = pic / max_val
            pic = pic.cpu().detach().numpy() * 255
            pic = pic.astype(np.uint8)
            ret1, pic = cv2.threshold(pic, 254, 255, cv2.THRESH_BINARY)
            dst = cv2.bitwise_and(frame, frame, mask=pic)
            #cv2.imshow("Target", dst)
            print(f"{i}: {num} Attention Loss: {lossc.item()} Trigger Loss: {lossc1.item()}", end="\r")
        #scheduler.step()
        #scheduler1.step()
        if accuracy/n*90 > max_acc:
            max_acc = accuracy/n*90
            max_acc_epoch = k
        if accuracy_tr/n*90 > max_acc_tr:
            max_acc_tr = accuracy_tr/n*90
            max_acc_epoch_tr = k
        print(f"Epoch {k}\n Attention Loss: {loss/n} Attention Accuracy: {accuracy/n*90}\n Trigger Loss: {loss_tr/n} Trigger Accuracy: {accuracy_tr/n*90}\n Best Attention Accuracy: {max_acc} Epoch {max_acc_epoch}\n Best Trigger Accuracy: {max_acc_tr} Epoch {max_acc_epoch_tr}")
        torch.save(trigger_net.state_dict(), f"../models/iris_trigger_ac_fl_{k}_ce_4bn_15f.pth")
        #torch.save(axis_transformers.state_dict(), f"../models/iris_transformers_ac_{k}_ce_4bn_20f.pth")