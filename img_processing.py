import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import math


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return result

def contrast_roi(img, low, high):
    h, w = img.shape
    img_ = np.zeros(img.shape, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            temp = int((255 / (high - low)) * (img[y][x] - low))
            if temp > 255:
                img_[y][x] = 255
            elif temp < 0:
                img_[y][x] = 0
            else:
                img_[y][x] = temp

    return img_


def filter_roi_carpal_and_joint(roi):
    img = roi.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ###이진화
    # 마스크 생성을 위해, 밝기 강조한 Lab으로 이미지 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # 모폴로지
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)

    # 블러
    img = cv2.GaussianBlur(img, (15, 15), 0)

    # threshold 적용을 위해 Lab에서 Grayscale로 이미지 변환
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이진화
    ret, mask = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)

    # 컨투어
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    ###강조
    img = roi.copy()

    # 모폴로지
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)

    # contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.mean() <= 15:
        low = img.mean() * 1.5
        high = img.mean() * 1.6
    elif img.mean() <= 20:
        low = img.mean() * 1.5
        high = img.mean() * 1.8
    else:
        low = img.mean() * 1.5
        high = img.mean() * 2

    img = contrast_roi(img, low, high)

    # 컨투어
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

    # 마스크랑 비트 연산
    img = cv2.bitwise_and(img, mask)

    # 크기 표준화
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def filter_roi_fingers(roi):
    img = roi.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ###이진화
    # 마스크 생성을 위해, 밝기 강조한 Lab으로 이미지 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    # 이미지 노이즈 제거
    img = cv2.fastNlMeansDenoising(img, None, 5, 9, 15)

    # 모폴로지
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k2)

    # Grayscale로 이미지 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 평탄화
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(70, 70))
    dst1 = clahe.apply(img)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    img = cv2.morphologyEx(dst1, cv2.MORPH_TOPHAT, k2)

    # print(np.mean(img))
    # print(np.mean(img)*1.18)

    # 이진화
    threshold = 0
    if np.mean(img) > 12:
        threshold = 7
    elif 10.5 <= np.mean(img) <= 11.6:
        threshold = 6.5
    elif 10 <= np.mean(img) <= 10.4:
        threshold = 6.0
    elif 9.3 <= np.mean(img) <= 9.9:
        threshold = 5.8
    elif 8.3 <= np.mean(img) <= 9.2:
        threshold = 5.6
    elif 5.9 <= np.mean(img) <= 8.2:
        threshold = 5.0
    elif 3.8 <= np.mean(img) <= 5.8:
        threshold = 4.0
    elif np.mean(img) < 3.8:
        threshold = np.mean(img)

    # print(threshold)
    ret, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # 컨투어
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    ###강조
    img = roi.copy()

    # 모폴로지
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)

    # contrast
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.mean() <= 15:
        low = img.mean() * 1.5
        high = img.mean() * 1.6
    elif img.mean() <= 20:
        low = img.mean() * 1.5
        high = img.mean() * 1.8
    else:
        low = img.mean() * 1.5
        high = img.mean() * 2

    img = contrast_roi(img, low, high)

    # 컨투어
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), -1)

    # 마스크랑 비트 연산
    img = cv2.bitwise_and(img, mask)

    # 크기 표준화
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    roi_height, roi_width = img.shape

    for y in range(roi_height):
        for x in range(roi_width):
            if 0 <= y < roi_height * 0.5 and x > 220:
                img[y][x] = 0
            if roi_height * 0.5 <= y < roi_height * 0.70 and x > 243:
                img[y][x] = 0

    return img


def roi_finger_bone(img):
    img = cv2.resize(img, (700, 928))
    r_img_ = img.copy()
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_.shape
    img_ = img_[0:(int)(height * 0.9), 0:(int)(width * 0.95)]
    ret, img = cv2.threshold(img_, img.mean() * 0.99, 255, cv2.THRESH_BINARY)

    # 좌표 저장공간 설정(첫번째 공간에는 무게중심점, 두번째 공간에는 start점, 세번째에는 far점)
    dots = [[], [], []]

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt = max(contours, key=cv2.contourArea)

    mask = np.zeros(img.shape, dtype=np.uint8)

    # 컨투어 구하고 구한 컨투어로 이미지 그리기
    cv2.drawContours(mask, [max_cnt], -1, (255, 255, 255), -1)

    # 이미지 팽창
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, k)

    # 회전하기 위해서 가운데 손가락이 가장 길기때문에 가장 먼저 흰색을 나타내는 좌표를 찾아 낸다.
    first_255_x_point = 0
    first_255_y_point = 0

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)

    # 무게중심점 구하기 일단
    M = cv2.moments(max_cnt)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    for y, x_r in enumerate(mask):
        if 255 in x_r:
            x_255_indexs = np.where(x_r == 255)[0]

            x_255_mid_index = x_255_indexs[(int)(len(x_255_indexs) / 2)]
            first_255_x_point = x_255_mid_index

            first_255_y_point = y
            break

    # width 의 중간값의 x값과 가장 먼저 흰색을 나타내는 좌표의 x값을 비교한다.
    # 그리고 center 값을 기준으로 회전한다.

    half_h, half_w = center[1], center[0]
    ry = center[1] - first_255_y_point
    rx = abs(first_255_x_point - half_w)
    radian = math.atan2(ry, rx)
    degree = 90 - (radian * 180 / math.pi)
    #print("회전각도 : ", degree)

    if degree > 3:
        if first_255_x_point < half_w:
            mask = rotate_image(mask, 360 - degree)
            r_img_ = rotate_image(img_, 360 - degree)
        else:
            mask = rotate_image(mask, degree)
            r_img_ = rotate_image(img_, degree)

            # 이미지 외부 컨투어 구하기
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)

    # 무게중심점 구하기 일단
    M = cv2.moments(max_cnt)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    # cv2.circle(r_img_,center,0,[255,255,255],10)
    dots[0].append(center)

    # 볼록한 지점 구하기
    hull1 = cv2.convexHull(max_cnt)
    # cv2.drawContours(img_, [hull1], -1, (0,255,0),10)

    # 새끼손가락 시작점 x좌표
    little_finger_x = ()
    little_finger_y = ()

    min_little_finger_x = hull1[0][0][0]
    max_little_finger_y = hull1[0][0][1]

    for i in range(1, len(hull1)):
        if min_little_finger_x > hull1[i][0][0] and hull1[i][0][0] > 0 and hull1[i][0][1] < center[1]:
            min_little_finger_x = hull1[i][0][0]
            max_little_finger_y = hull1[i][0][1]
            little_finger_x, little_finger_y = hull1[i][0]

    # 엄지손가락 시작점 좌표
    thumb_x = ()
    thumb_y = ()

    max_thumb_x = hull1[0][0][0]
    for i in range(0, len(hull1)):
        if max_thumb_x < hull1[i][0][0] and hull1[i][0][1] < center[1] * 1.4 and hull1[i][0][1] > height * 0.3:
            max_thumb_x = hull1[i][0][0]
            thumb_x, thumb_y = hull1[i][0]

            # 볼록한 지점 구하기
    hull = cv2.convexHull(max_cnt, returnPoints=False)

    # 오목한 지점 구하기
    defects = cv2.convexityDefects(max_cnt, hull)

    # 거리를 저장 할 수 있는 공간 생성
    di = []

    for index in range(defects.shape[0]):
        # 시작점,끝점,far점,거리 할당
        start_point, end_point, far_point, distance = defects[index, 0]

        far = tuple(max_cnt[far_point][0])
        start = tuple(max_cnt[start_point][0])
        end = tuple(max_cnt[end_point][0])

        # cv2.circle(r_img_,far,2,[255,255,255],10)
        # cv2.circle(r_img_,start,2,[255,255,255],10)

        # cv2.circle(r_img_,end,2,[255,255,255],10)
        # 거리를 저장
        di.append(distance)
        dots[1].append(start)
        dots[2].append(far)

    # 더 쉽게 불러오기 위해서 far, start로 분리
    far_pt = np.array(dots[2])
    start_pt = np.array(dots[1])

    far_xrange = []
    far_yrange = []
    far_miny = 1000
    far_maxy = 0
    start_miny = 1000
    # 가장 오목하게 들어가 있는 부분을 찾기 위해서 sort(내림차순)
    di = np.array(di)
    s_di = np.sort(di)[::-1]
    # 내림차순된 거리들을 6개만 뽑아내기 위해서 slice
    for i in list(s_di[:6]):
        index = np.where(di == i)[0]
        # 6개의 좌표들 중에서 가장 최저의 y 값을 찾는다. (손목쪽 roi에 필요)
        far_miny = min(far_miny, far_pt[index[0]][1])
        # 6개의 좌표들 중에서 가장 최고의 y 값을 찾는다.
        far_maxy = max(far_maxy, far_pt[index[0]][1])

        # 가장 오목한 지점 6개의 좌표를 출력
        # cv2.circle(r_img_,np.array(far_pt[index[0]]),2,[255,255,255],5)

        # 좌표들이 x,y로 나눠져 있어서 쉽게 비교하기 위해서 x,y끼리 나눈다.
        far_xrange.append(far_pt[index[0]][0])
        far_yrange.append(far_pt[index[0]][1])

    # far_xrange를 오름차순으로 정렬
    sorted_far_xrange = np.sort(far_xrange)
    sorted_far_yrange = np.sort(far_yrange)

    carpus_start_point = ((int)(np.sort(far_xrange)[0]), (int)(center[1] * 1.05))

    carpus_end_point_y = (int)(far_maxy * 1.09)
    carpus_end_point_x = np.sort(far_xrange)[-2]
    if carpus_end_point_x > center[0] * 1.4:
        carpus_end_point_x = np.sort(far_xrange)[-2]
    if carpus_end_point_y > height * 0.82:
        carpus_end_point_y = height * 0.82
    carpus_end_point = (carpus_end_point_x, int(carpus_end_point_y))

    # 새끼손가락 endpoint x 좌표
    little_finger_endpoint_x_list = []
    for x, y in zip(far_xrange, far_yrange):
        if y < int(center[1]):
            little_finger_endpoint_x_list.append(x)

    # 엄지손가락 endpoint
    thumb_endpoint_list = []
    for x, y in zip(far_xrange, far_yrange):
        if y > int(center[1]) and x > int(center[0]):
            thumb_endpoint_list.append([x, y])

    thumb_endpoint = []
    max_y = thumb_endpoint_list[0][1]
    # print(thumb_endpoint_list[0][1])
    if len(thumb_endpoint_list) == 1:
        thumb_endpoint = thumb_endpoint_list
    for i in range(1, len(thumb_endpoint_list)):
        if max_y < thumb_endpoint_list[i][1]:
            max_y = thumb_endpoint_list[i][1]
            thumb_endpoint.append(thumb_endpoint_list[i])
        else:
            thumb_endpoint.append(thumb_endpoint_list[0])

    # 손목뼈 부분 roi 를 하기 위해서는 가장 오목하게 들어가 있는 부분중에서 가장 최저 x값(xrange[0])과 center값의 y값을 시작점으로
    # 끝점으로는 가장 오목하게 들어가 있는 부분들 중에서 가장 최고 x값(xrange[-1])과 오목하게 들어간 점중에서 가장 최고 y값을 준다.
    # cv2.rectangle(r_img_,carpus_start_point,carpus_end_point,[255,255,255],5)
    wrist_roi = r_img_[(int)(center[1] * 1.05):int(carpus_end_point_y),
                (int)(np.sort(far_xrange)[0]):carpus_end_point_x]

    #print("==================================================================")
    #print("손목 관절 roi 추출 시작좌표 : ", carpus_start_point)
    #print("손목 관절 roi 추출 끝 좌표 : ", carpus_end_point)

    # 손목뼈 위쪽에 있는 관절 4개를 추출하기 위해서는 오목하게 들어가 있는 부분중에서 가장 최저 x값(xrange[0])과 far_miny 값을 y값으로
    # 끝점으로는 가장 오목하게 들어가 있는 부분들 중에서 가장 최고 x값(xrange[-1])과 center의 y값을 준다.

    four_end_point_x = np.sort(far_xrange)[-1]
    if four_end_point_x > center[0] * 1.4:
        four_end_point_x = np.sort(far_xrange)[-2]

    four_start_point_y = far_miny
    if four_start_point_y < int(little_finger_y * 0.8) and four_start_point_y < int(thumb_y * 0.9):
        four_start_point_y = int(center[1] * 0.7)

    four_start_point = ((int)(np.sort(far_xrange)[0] * 0.85), four_start_point_y)
    four_end_point = (four_end_point_x, (int)(center[1] * 1.05))

    # cv2.rectangle(r_img_,four_start_point,four_end_point,[255,255,255],5)

    middle_roi = r_img_[four_start_point_y:(int)(center[1] * 1.05),
                 (int)(np.sort(far_xrange)[0] * 0.85):four_end_point_x]

    #print("==================================================================")
    #print("가운데 4개 관절 roi 추출 시작좌표 : ", four_start_point)
    #print("가운데 4개 관절 roi 추출 끝 좌표 : ", four_end_point)
    # 가운데 손가락을 추출하기 위해서 start_pt의 x 좌표가 sorted_far_xrange 에서 3번째와 4번째 값 사이에 있어야 한다.
    middle_index = np.where((start_pt[:, 0] <= (int)(sorted_far_xrange[3])) & (start_pt[:, 0] >= sorted_far_xrange[2]))[
        0]
    # 위에 조건식으로 나온 인덱스를 start_point에 대입하면 만족하는 좌표들이 여러개 나올것이다.
    middle_points = start_pt[middle_index]

    # 새끼손가락 ROI
    little_finger_end_point_x = min(little_finger_endpoint_x_list)
    if carpus_start_point[0] == little_finger_end_point_x:
        little_finger_end_point_x = np.sort(little_finger_endpoint_x_list)[1]
    if little_finger_end_point_x == sorted_far_xrange[2]:
        little_finger_end_point_x = min(little_finger_endpoint_x_list)

    little_finger_start_point = (int(little_finger_x * 0.5), int(little_finger_y * 0.8))
    little_finger_end_point = (little_finger_end_point_x, (int)(center[1] * 1.05))
    # cv2.rectangle(r_img_,little_finger_start_point, little_finger_end_point,[255,255,255],2)

    little_finger_roi = r_img_[int(little_finger_y * 0.8):(int)(center[1] * 1.05),
                        int(little_finger_x * 0.5):little_finger_end_point_x]
    # cv2.imshow('little_finger_roi', little_finger_roi)
    #print("==================================================================")
    #print("새끼손가락 roi 추출 시작좌표 : ", little_finger_start_point)
    #print("새끼손가락 roi 추출 끝 좌표 : ", little_finger_end_point)

    # 엄지손가락 ROI

    if not thumb_x or not thumb_y:
        thumb_x, thumb_y = 610, 250
        thumb_end_point_x, thumb_end_point_y = 450, 619
    else:
        thumb_x, thumb_y = int(thumb_x * 1.02), int(thumb_y * 0.9)
        thumb_end_point_x, thumb_end_point_y = int(thumb_endpoint[0][0] * 1.05), int(thumb_endpoint[0][1])

    thumb_start_point = (thumb_x, thumb_y)
    thumb_end_point = (thumb_end_point_x, int(thumb_end_point_y))

    # cv2.rectangle(r_img_,thumb_start_point, thumb_end_point,[255,255,255],2)

    thumb_roi = r_img_[thumb_y:thumb_end_point_y, thumb_end_point_x:thumb_x]
    # cv2.imshow('thumb_roi', thumb_roi)
    #print("==================================================================")
    #print("엄지손가락 roi 추출 시작좌표 : ", thumb_start_point)
    #print("엄지손가락 roi 추출 끝 좌표 : ", thumb_end_point)

    for point in middle_points:
        # 가운데 손가락 사이에 있는 좌표들 중에서 최저 y 값을 찾는다.
        start_miny = min(start_miny, point[1])

    start_maxy = max(far_yrange[np.where(far_xrange == sorted_far_xrange[3])[0][0]],
                     far_yrange[np.where(far_xrange == sorted_far_xrange[2])[0][0]])
    if start_maxy > center[1]:
        start_maxy = center[1] * 0.8
    # 시작 좌표로는 x값으로 sorted_far_xrange 에서 3번째와 y 값으로는 최저y 값을 주고
    # 마지막 좌표로는 x값으로 sorted_far_xrange 에서 4번째와 y값으로는 3번째 와 4번째 좌표의 최고 y 값을 준다.
    middle_finger_start_point = (sorted_far_xrange[2], start_miny)
    middle_finger_end_point = (sorted_far_xrange[3], int(start_maxy * 1.1))

    # cv2.rectangle(r_img_,middle_finger_start_point,middle_finger_end_point,[255,255,255],5)

    middle_finger_roi = r_img_[start_miny:int(start_maxy * 1.1), sorted_far_xrange[2]: sorted_far_xrange[3]]
    # cv2.imshow('middle_finger_roi', middle_finger_roi)
    #print("==================================================================")
    #print("가운데 손가락 roi 추출 시작좌표 : ", middle_finger_start_point)
    #print("가운데 손가락 roi 추출 끝 좌표 : ", middle_finger_end_point)

    rois = np.array([wrist_roi, middle_roi, little_finger_roi, thumb_roi, middle_finger_roi])
    # cv2.imshow('r_img_',r_img_)
    return rois

def img_roi(img):
    rois = roi_finger_bone(img)
    for j in range(0, len(rois)):
        if j == 0:
            wrist_roi = filter_roi_carpal_and_joint(rois[j])
    return wrist_roi

def print_roi(img):
    rois = roi_finger_bone(img)
    try:
        for j in range(0, len(rois)):
            if j == 0:
                wrist_roi = filter_roi_carpal_and_joint(rois[j])
                if wrist_roi.ndim == 3:
                    wrist_roi = wrist_roi[:, :, 0]
            elif j == 1:
                middle_roi = filter_roi_carpal_and_joint(rois[j])
                if middle_roi.ndim == 3:
                    middle_roi = middle_roi[:, :, 0]
            elif j == 2:
                little_finger_roi = filter_roi_fingers(rois[j])
                if little_finger_roi.ndim == 3:
                    little_finger_roi = little_finger_roi[:, :, 0]
            elif j == 3:
                thumb_roi = filter_roi_fingers(rois[j])
                if thumb_roi.ndim == 3:
                    thumb_roi = thumb_roi[:, :, 0]
            else:
                middle_finger_roi = filter_roi_fingers(rois[j])
                if middle_finger_roi.ndim == 3:
                    middle_finger_roi = middle_finger_roi[:, :, 0]
        roi_ = np.array([wrist_roi, middle_roi, little_finger_roi, thumb_roi, middle_finger_roi])
        return roi_
    except:
        print("에러처리")
        pass