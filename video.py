import os
import cv2

import numpy as np
#图片路径
path = "/nfs/volume-382-121/ivytang_i/codes/StreamMapNet/demo/scene-0010/"
#path = '/data/txw/snowy/ptest11/'
#输出视频路径
video_dir = '/nfs/volume-382-121/ivytang_i/codes/StreamMapNet/demo0010.mp4'
#帧率
fps = 10
#图片尺寸
img_size = (1680,450)#(6000,1800)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #opencv3.0cv2.VideoWriter_fourcc('M', 'P', '4', '2')('D','I','V','X')
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size, True)
for i in range(1,40):
    pred = cv2.imread(path +str(i)+'/pred/map.jpg')
    gt = cv2.imread(path +str(i)+'/gt/map.jpg')
    rotate_pred = cv2.rotate(pred, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotate_gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotate_pred = cv2.copyMakeBorder(rotate_pred, 10, 10, 5, 10, cv2.BORDER_CONSTANT, None, value=0)
    rotate_gt = cv2.copyMakeBorder(rotate_gt, 10, 10, 10, 5, cv2.BORDER_CONSTANT, None, value=0)

    cam_front = cv2.imread(path +str(i)+'/gt/CAM_FRONT.jpg')
    cam_front_l = cv2.imread(path + str(i) + '/gt/CAM_FRONT_LEFT.jpg')
    cam_front_r = cv2.imread(path + str(i) + '/gt/CAM_FRONT_RIGHT.jpg')
    cam_back = cv2.imread(path + str(i) + '/gt/CAM_BACK.jpg')
    cam_back_l = cv2.imread(path + str(i) + '/gt/CAM_BACK_LEFT.jpg')
    cam_back_r = cv2.imread(path + str(i) + '/gt/CAM_BACK_RIGHT.jpg')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rotate_pred, "PRED", (10, 60), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(rotate_gt, "GT", (10, 60), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(cam_front, "CAM_FRONT", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cam_front_l, "CAM_FRONT_LEFT", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cam_front_r, "CAM_FRONT_RIGHT", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cam_back, "CAM_BACK", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cam_back_l, "CAM_BACK_LEFT", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cam_back_r, "CAM_BACK_RIGHT", (10, 60), font, 2, (255, 255, 255), 5, cv2.LINE_AA)

    gt_pred = np.concatenate([rotate_gt, rotate_pred], axis=1)  # axis=0纵向  axis=1横向
    gt_pred = cv2.resize(gt_pred, (1800, 1800))
    #cv2.imwrite(path + str(i) + 'test.jpg', gt_pred)
    front_image = np.concatenate([cam_front_l, cam_front,cam_front_r], axis=1)
    back_image = np.concatenate([cam_back_l, cam_back, cam_back_r], axis=1)
    cam_image = np.concatenate([front_image, back_image], axis=0)

    final_vis_image = np.concatenate([cam_image, gt_pred], axis=1)
    #import pdb
    #pdb.set_trace()
    if i<10:
        timestamp = '0'+ str(i)
    else:
        timestamp = str(i)
    cv2.imwrite(path+'all_vis_'+ timestamp+ '.jpg',final_vis_image)
    cv2.imwrite(path+'camera_'+ timestamp+ '.jpg',cam_image)
    resized_img = cv2.resize(final_vis_image, img_size)

    videoWriter.write(resized_img)

videoWriter.release()
print('finish')

