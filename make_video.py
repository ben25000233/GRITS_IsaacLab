import numpy as np
import cv2
import os


def draw_text_with_bg(img, text, pos, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 1
    margin = 5

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - margin, y - text_size[1] - margin),
                  (x + text_size[0] + margin, y + margin),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

def draw_spillage_overlay(img, prob_before, prob_guided):

    overlay = img.copy()
    h, w = img.shape[:2]
    x, y = 580, 530
    bar_width = 300
    bar_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # ----------- text ------------
    text1 = "{:.4f}".format(prob_guided)
    text2 = "{:.4f}".format(prob_before)
    text3 = "Spillage Probability"
    cv2.putText(img, text1, (x + 450, y + 40), font, 1., (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x + 450, y + 90), font, 1., (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, text3, (x + 130, y - 10), font, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # ----------- bar ------------
    bx = x + 130
    by1 = y + 22
    by2 = y + 60
    bar_len1 = int(prob_guided * bar_width)
    bar_len2 = int(prob_before * bar_width)
    cv2.rectangle(img, (bx, by1), (bx + bar_len1, by1 + bar_height), (255, 0, 0), -1) # blue
    cv2.rectangle(img, (bx, by2), (bx + bar_len2, by2 + bar_height), (0, 0, 255), -1) # red    
    cv2.rectangle(img, (bx, by1), (bx + bar_width, by1 + bar_height), (255, 255, 255), 2) # blue
    cv2.rectangle(img, (bx, by2), (bx + bar_width, by2 + bar_height), (255, 255, 255), 2) # red    

    return img



food_list = [
            'mkh'
            'bdh', 'bdl', 
            'fuh', 'ful',
            'mah', 'mal', 
            'mth', 'mtl', 
            'nuh', 'nul', 
            'peh', 'pel', 
            'rbh', 'rbl', 
            'sah', 'sal', 
            'snh', 'snl'
             ]


for food_name in food_list:
    print(food_name)

    # get all image
    os.makedirs('video_success/914/{}/denoise/all'.format(food_name), exist_ok=True)
    rgb_npy = np.load('video_success/914/{}/back_rgb_video.npy'.format(food_name))
    frame = 1 
    for k in range(rgb_npy.shape[0]):
        for m in range(rgb_npy.shape[1]):
            rgb = cv2.cvtColor(rgb_npy[k, m, :, :, :], cv2.COLOR_BGR2RGB)
            cv2.imwrite('video_success/914/{}/denoise/all/{}.png'.format(food_name, str(frame).zfill(3)), rgb)
            frame += 1

    # out = cv2.VideoWriter('/home/yling/Desktop/GRITS/demo/ppt/{}_success.mp4'.format(food_name), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 960))
    # for i in range(24, 72):
    #     img = cv2.imread('prob_video_with_guided/{}_denoise/all/{}.png'.format(food_name, str(i+1).zfill(3)))
    #     out.write(img)
    # out.release()

    # to video
    out = cv2.VideoWriter('video_success/914/{}/{}.mp4'.format(food_name, food_name), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 960))
    prob_before = np.load('video_success/914/{}/spillage_prob_before.npy'.format(food_name))
    prob_guided = np.load('video_success/914/{}/spillage_prob.npy'.format(food_name))
    start = 3
    end = 7
    k = 12*start
    for i in range(start, end):
        for j in range(6):
            img = cv2.imread('video_success/914/{}/denoise/both/{}/{}.png'.format(food_name, i+1, str(j+1).zfill(3)))
            out.write(img)
            if j==5:
                for _ in range(20):               
                    img_copy = img.copy()
                    img_with_overlay = draw_spillage_overlay(img_copy, prob_before[i], prob_guided[i])
                    out.write(img_with_overlay)
        # execute 12 frame
        for _ in range(12):
            img_execute = cv2.imread('video_success/914/{}/denoise/all/{}.png'.format(food_name, str(k).zfill(3)))
            out.write(img_execute)
            k += 1
        if i==5:
            for _ in range(3):
                out.write(img_execute)
    out.release()

