import json
import os
import threading

import cv2
import numpy as np
from ultralytics import YOLO

width = 1920
height = 1080

datamap = {'car': 0, 'truck': 1, 'bus': 2}
ds_path = 'ds/'


def _ret2abs(_item):
    for i in range(2):
        _item[i][0] = int(_item[i][0] * width)
        _item[i][1] = int(_item[i][1] * height)
    return _item


def get_borders(_data):
    _near_in = _ret2abs((_data['zones'][0][0].copy(), _data['zones'][0][3].copy()))
    _near_out = _ret2abs((_data['zones'][1][0].copy(), _data['zones'][1][3].copy()))
    _right_out_border = _ret2abs((_data['areas'][0][0].copy(), _data['areas'][0][1].copy()))

    if len(_data['areas']) > 1:
        _left_out_border = _ret2abs((_data['areas'][1][2].copy(), _data['areas'][1][3].copy()))
    else:
        _left_out_border = None

    if len(_data['zones']) > 2:
        _far_in = _ret2abs((_data['zones'][2][0].copy(), _data['zones'][2][3].copy()))
        _far_out = _ret2abs((_data['zones'][3][0].copy(), _data['zones'][3][3].copy()))
    else:
        _far_in = None
        _far_out = None

    return _near_in, _near_out, _far_in, _far_out, _left_out_border, _right_out_border


def draw_l_lines(_img, l_lines):
    near_in, near_out, far_in, far_out, left_out_border, right_out_border = l_lines

    cv2.line(_img, *near_in, color=(0, 255, 0), thickness=3)
    cv2.line(_img, *near_out, color=(0, 0, 255), thickness=3)

    cv2.line(_img, *far_in, color=(0, 255, 0), thickness=3)
    cv2.line(_img, *far_out, color=(0, 0, 255), thickness=3)

    cv2.line(_img, *left_out_border, color=(0, 127, 255), thickness=3)
    cv2.line(_img, *right_out_border, color=(0, 127, 255), thickness=3)

    return _img


def check_out_borders(box, limit_lines):
    c_x, c_y = box

    near_in, near_out, far_in, far_out, left_out_border, right_out_border = limit_lines
    if far_in is not None:
        if c_x > far_in[1][0]:
            return 0
    elif c_x > near_out[1][0]:
        return 0
    if c_x < near_in[0][0]:
        return 0
    if c_y > near_out[0][1]:
        return 0
    if far_out is not None:
        if c_y < far_out[1][1]:
            return 0
    elif c_y < near_in[1][1]:
        return 0

    return 1


def run_tracker_in_thread(filename, model, gl):
    results_map = {
        'car_cnt': 0,
        'truck_cnt': 0,
        'bus_cnt': 0,
        'car_vlc': [],
        'truck_vlc': [],
        'bus_vlc': [],
    }
    json_file = f"ds/markup/jsons/{filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.json"
    with open(json_file, 'r') as rf:
        data = json.load(rf)
    limit_lines = get_borders(data)

    tracking_dict = {}

    cap = cv2.VideoCapture(filename)
    count = 0
    while True:
        success, frame = cap.read()
        count += 11  # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

        if not success:
            break

        frame_result = model.track(frame, persist=True, verbose=False)[0]
        frame_data = frame_result.boxes
        boxes = frame_data.xyxy.cpu().numpy().astype(np.int_)
        clss = frame_data.cls.detach().cpu().numpy().astype(np.int_)
        try:
            ids = frame_data.id.detach().cpu().numpy().astype(np.int_)
        except Exception as e:
            print(e)
            continue

        for (exp_box, exp_cls, exp_id) in zip(boxes, clss, ids):
            cls_name = frame_result.names[exp_cls]
            tl_x, tl_y, rb_x, rb_y = exp_box
            c_x = ((rb_x - tl_x) // 2) + tl_x
            c_y = ((tl_y - rb_y) // 2) + rb_y

            if cls_name not in datamap:
                continue
            check_in_area = check_out_borders((c_x, c_y), limit_lines)
            if check_in_area and exp_id not in tracking_dict:
                tracking_dict[exp_id] = [count, count, [cls_name]]

            elif check_in_area and exp_id in tracking_dict:
                if count - tracking_dict[exp_id][1] < 29:
                    tracking_dict[exp_id][1] = count
                    tracking_dict[exp_id][2].append(cls_name)
                else:
                    tracking_dict[f"{exp_id}_{count}"] = tracking_dict.pop(exp_id)
                    tracking_dict[exp_id] = [count, count, [cls_name]]

    for e_id, (first_frame_n, last_frame_n, cls_list) in tracking_dict.items():
        if first_frame_n == last_frame_n:
            continue
        sum_cls_name = max(set(cls_list), key=cls_list.count)
        results_map[f"{sum_cls_name}_cnt"] += 1
        velocity = 20 / ((last_frame_n - first_frame_n) / 29)
        results_map[f"{sum_cls_name}_vlc"].append(velocity * 3.6)
    for key in results_map.keys():
        if key.endswith('vlc'):
            if len(results_map[key]) > 0:
                results_map[key] = np.median(results_map[key])
            else:
                results_map[key] = 0

    gl[filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]] = results_map


threads_num = 32
rev = False

all_videos = 'ds/all_videos/'
if not os.path.exists(all_videos):
    os.mkdir(all_videos)

for path in os.listdir(ds_path):
    if not os.path.isdir(ds_path + path) or path == 'markup':
        continue
    s_path = f"{ds_path}{path}/"
    for file in os.listdir(s_path):
        if file.endswith('.mp4'):
            os.rename(s_path + file, all_videos + file)

all_video_files = os.listdir(all_videos)

if not rev:
    b_range = range(len(all_video_files) // threads_num + 1)
else:
    b_range = range(len(all_video_files) // threads_num + 1, -1, -1)

for batch_n in b_range:
    glob_results = {}
    video_files = [
        f"{all_videos}{item}"
        for item in all_video_files[batch_n * threads_num: (batch_n + 1) * threads_num]
    ]
    models = [YOLO('yolov8x.pt') for _ in range(threads_num)]
    if len(video_files) != threads_num:
        threads_num = len(video_files)

    threads = [
        threading.Thread(
            target=run_tracker_in_thread,
            args=(video_files[i], models[i], glob_results),
            daemon=True
        )
        for i in range(threads_num)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    s = ''
    for k, v in glob_results.items():
        rr = [
            k,
            'car', v['car_cnt'], v['car_vlc'],
            'van', v['truck_cnt'], v['truck_vlc'],
            'bus', v['bus_cnt'], v['bus_vlc']
        ]
        s += ','.join([str(item) for item in rr]) + '\r'

    with open('result.csv', 'a') as f:
        f.write(s)
