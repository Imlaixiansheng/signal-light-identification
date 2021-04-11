import cv2 as cv
import numpy as np
import os
import sys


def key_000_line(str="result"):
    print("*" * 40, str, "*" * 40)


# 创建活动窗口(按img原始比例显示)
def key_000_point(img, str="result", win_width=800):
    print(str, "的shape:", img.shape)
    ratio = img.shape[0] / img.shape[1]
    cv.namedWindow(str, cv.WINDOW_KEEPRATIO)
    cv.moveWindow(str, 20, 20)
    cv.resizeWindow(str, win_width, int(win_width * ratio))
    cv.imshow(str, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 显示文字信息的图片
def message_box(str, h=100, w=550):
    img = np.zeros((h, w, 3), np.uint8)
    cv.putText(img, str, (50, int(h / 2)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 返回显示文字信息的图片
def message_box_ret(str, h=100, w=550):
    img = np.zeros((h, w, 3), np.uint8)
    cv.putText(img, str, (50, int(h / 2)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    return img


# 按原始比例对图像进行缩放（resize后的大小对后面识别有很大影响）
def resize_img(img, long_side=2000):
    cols, rows = img.shape[:2]
    print("原始图像的高，宽，数据类型：", cols, rows, img.dtype)
    if cols < rows:
        new_img = cv.resize(img, (long_side, int(long_side / rows * cols)), interpolation=cv.INTER_CUBIC)
    else:
        new_img = cv.resize(img, (int(long_side / cols * rows), long_side), interpolation=cv.INTER_CUBIC)
    print("resize之后图像的高，宽，数据类型：", new_img.shape[0], new_img.shape[1], img.dtype)
    return new_img


# 对图像进行不同色彩空间的转换
def color_space_demo(image, choice=1):
    # 将RGB图像转到HSV色彩空间
    img_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 将RGB图像转到HIS色彩空间
    img_HIS = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    if choice == 1:
        return img_HSV
    else:
        return img_HIS


# def contrast_brightness_damo(image, c, b):
#     h, w = image.shape
#     blank = np.zeros([h, w], image.dtype)
#     dst = cv.addWeighted(image, c, blank, 1 - c, b)
#     return dst
#
#
#
# def mean_blur_demo(src):
#     mean_blur = cv.blur(src, (5, 5))
#     # src:要处理的图像
#     # (5，5):卷积核大小
#     return mean_blur
#
#
# def median_blur_demo(src):
#     median_blur = cv.medianBlur(src, 5)
#     # src:要处理的图像
#     # 5: 卷积核大小（只能取奇数）
#     return median_blur


# 逐个显示轮廓
def show_every_cnts(cnts):
    for cnt in cnts:
        img_contours = cv.drawContours(img.copy(), cnt, -1, (255, 0, 255), 4)
        key_000_point(img_contours)


# 轮廓圆形度筛选
def cnts_yuanxingdu(img, k=0.7):
    cnts_yxd_k = []
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i, cnt in zip(range(len(contours)), contours):
        area = cv.contourArea(cnt)  # 获取轮廓面积
        perimeter = cv.arcLength(cnt, True)  # 获取轮廓周长
        if perimeter > 0:
            F = 4 * np.pi * area / perimeter / perimeter  # 根据圆形度公式求轮廓圆形度（之前求得8个轮廓的圆形度均大于0.75，这里将阈值设为0.7）
            if F >= k:
                cnts_yxd_k.append(cnt)
    print("opening处理完的mask的圆形度大于", k, "的轮廓数量：", len(cnts_yxd_k))
    return cnts_yxd_k


# # 轮廓近似
# def contours_jinsi(contours, k=0.01):
#     approx = []
#     for i, cnt in zip(range(len(contours)), contours):
#         cnt = contours[i]
#         yuzhi = k * cv.arcLength(cnt, True)  # 先设定阈值，一般取轮廓周长的0.1倍（阈值越小，越不忽略毛刺，阈值=0时，跟正常轮廓一致）
#         approx.append(cv.approxPolyDP(cnt, yuzhi, True))  # 求近似轮廓，将“初始轮廓”和“所设阈值”传进去，得到新的近似轮廓
#     return approx

path = os.path.abspath(os.path.dirname(sys.argv[0]))
panel = cv.imread(path+"/images_box/light_panel.jpg")
template = cv.imread(path+"/images_box/template_split_line2000.png")
# resize+gs一下(resize后的大小对后面识别有很大影响)
img = resize_img(panel, long_side=2000)
img_gs = cv.GaussianBlur(img, (9, 9), 0)
key_000_point(img, str="origial")
# gray色彩空间
img_gray = cv.cvtColor(img_gs, cv.COLOR_RGB2GRAY)
# canny边缘检测(不需要再二值化)
canny = cv.Canny(img_gray, 60, 65)
key_000_point(canny, str="canny")
# 端点连接处理？？？？？？
# 圆形度检测
cnts_circle_level = cnts_yuanxingdu(canny, k=0.85)
img_circle_level = cv.drawContours(img.copy(), cnts_circle_level, -1, (255, 0, 255), 5)
key_000_point(img_circle_level, "img_circle_level")
# 设定获取颜色的阈值(此处捕获的是紫色，捕获的是上一步自己画的轮廓，阈值可以给的狠一些)
img_hsv = cv.cvtColor(img_circle_level, cv.COLOR_RGB2HSV)
color_lower = np.array([145, 250, 250])
color_upper = np.array([155, 255, 255])
mask = cv.inRange(img_hsv, color_lower, color_upper)
key_000_point(mask, str="mask")
# 对捕捉到的mask进行轮廓检测（线宽 = -1）
cnts_mask, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
img_mask_cnts = cv.drawContours(img.copy(), cnts_mask, -1, (255, 0, 255), -1)
key_000_point(img_mask_cnts, "img_mask_cnts")
# 对圆形度+mask颜色捕捉筛选完的轮廓进行面积排序（找面积阈值,resize的尺寸不同这个阈值也不同）
cnts_circle_area = sorted(cnts_mask, key=cv.contourArea, reverse=True)[:10]
img_area_level = cv.drawContours(img.copy(), cnts_circle_area, -1, (255, 0, 255), -1)
key_000_point(img_area_level, "img_area_level")
# 轮廓面积
for i, cnt_area in zip(range(len(cnts_circle_area)), cnts_circle_area):
    area = cv.contourArea(cnts_circle_area[i])
    print("第", i + 1, "个轮廓的面积为：", area)
# 把当前得到的圆形轮廓抠出来（变成一组roi，以便进行后面的模板匹配）
roi_group = []
circle_rectangle_point = []
for i, c in zip(range(len(cnts_circle_area)), cnts_circle_area):
    x, y, w, h = cv.boundingRect(c)
    circle_rectangle_point.append([x, y, w, h])  # 把圆形的外接矩形坐标存下来
    roi = img[y:y + h, x:x + w]  # 把外接矩形作为哦roi区域抠出来
    # roi = cv.resize(roi,(100,100))
    # cv.imwrite("1.png", roi)
    roi_group.append(roi)
    # key_000_point(roi, win_width=100)
# 按x进行排序：将roi和外接矩形坐标x组成的字典
list_cle_pit = np.array(circle_rectangle_point)
list_pit_x = list_cle_pit[:, 0]  # 取到圆形外接矩形坐标x值
dict_x_roi = dict(zip(list_pit_x, roi_group))  # 将x坐标-roi组成字典
list_x_roi = list(dict_x_roi.items())  # 字典转列表（方便排序）
list.sort(list_x_roi, key=lambda x: x[0])  # 排序（list_pit_roi是一个将roi按x坐标从左到右排序的列表）
# 按x进行排序：外接矩形坐标x和其形状信息组成的字典
list_tangle_shape = list_cle_pit[:, 1:4]
dict_x_shape = dict(zip(list_pit_x, list_tangle_shape))
# list_x_shape = list(dict_x_shape.items())
# list.sort(list_x_shape, key=lambda x: x[0])
# 模板匹配(用roi_group[i]和template进行匹配)(roi_group[i]需要resize成100*100)
key_000_point(template, str="template")
message_box("START_MATCH!!!")
for i in range(len(list_x_roi)):
    roi_match = cv.resize(list_x_roi[i][1], (100, 100))  # 这个地方resize可能会让很小的误差被放大（设置面积阈值）
    # key_000_point(roi_match, str="roi_match", win_width=100)
    match_torch = cv.matchTemplate(template, roi_match, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_torch)  # 获取匹配位置坐标
    print("第", i, "个roi匹配得的min_val, max_val, min_loc, max_loc", min_val, max_val, min_loc, max_loc)
    # 在原图中显示一下当前的roi（用方框标记出来）
    target_x, [target_y, target_w, target_h] = list_x_roi[i][0], dict_x_shape[list_x_roi[i][0]]
    print("target_x,target_y,target_w,target_h", target_x, target_y, target_w, target_h)
    top_left = (target_x, target_y)
    bottom_right = (top_left[0] + target_w, top_left[1] + target_h)
    res = cv.rectangle(img.copy(), top_left, bottom_right, (255, 0, 255), 3)  # 在img_copy图像上画矩形
    key_000_point(res, "res")
    # 用这个最大值最小值判断一下，有些不是信号灯的图，看能否筛选掉（把match_torch的矩阵值拿出来看看）（考虑是否用TM_CCORR_NORMED）
    if max_val < 0.85:
        message_box("No Match!!!")
        key_000_line()
    else:
        # 根据max_loc[0]位置坐标来判断是哪个灯
        if max_loc[0] <= 9:
            # print("识别结果：模板1号")
            msg_img = message_box_ret("template_number_01!!!")
        elif 140 <= max_loc[0] <= 160:
            # print("识别结果：模板2号")
            msg_img = message_box_ret("template_number_02!!!")
        elif 290 <= max_loc[0] <= 310:
            # print("识别结果：模板3号")
            msg_img = message_box_ret("template_number_03!!!")
        elif 440 <= max_loc[0]:
            # print("识别结果：模板4号")
            msg_img = message_box_ret("template_number_04!!!")
        top_left = max_loc  # 此处要根据所选择的算法，确认原点坐标是最大值还是最小值！
        h, w = roi_match.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        res = cv.rectangle(template.copy(), top_left, bottom_right, (255, 0, 255), 2)  # 在img_copy图像上画矩形
        res = np.vstack((res, msg_img))
        key_000_point(res, "res")
        key_000_line()
message_box("Finish!!!")
