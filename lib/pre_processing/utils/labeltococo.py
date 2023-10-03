'''
datasets process for object detection project.
for convert customer dataset format to coco data format,
'''

import traceback
import argparse
import datetime
import json
import cv2
import os
import json
import numpy as np
__CLASS__ = ['__background__', 'car']   # class dictionary, background must be in first index.

import json


class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)



def argparser():
    parser = argparse.ArgumentParser("define argument parser for pycococreator!")
    #图片和标签的路径
    parser.add_argument("-r", "--project_path", default="/home/cidi/桌面/Keypoint/heatmap/data/07_24/train", help="path of root directory")
    parser.add_argument("-l", "--label_path", default="/home/cidi/桌面/Keypoint/heatmap/data/07_24/train/labels",
                        help="path of root directory")
    #图片所在的文件夹，可以支持多个文件夹
    parser.add_argument("-p", "--phase_folder", default="/home/cidi/桌面/Keypoint/heatmap/data/07_24/train/imgs", help="datasets path of [train, val, test]")

    #关键点个数
    parser.add_argument("-joint", "--pointNum", default=16,help="point numbers")
    # 是否开启关键点读取，
    parser.add_argument("-po", "--have_points", default=True, help="if have points we will deal it!")
    # 是否开启把点画在图片上验证读取，
    parser.add_argument("-check", "--have_check", default=True, help="if you want to check all points!")

    return parser.parse_args()

def MainProcessing(args):
    '''main process source code.'''
    annotations = {}    # annotations dictionary, which will dump to json format file.
    project_path = args.project_path
    phase_folder = args.phase_folder
    # coco annotations info.
    annotations["info"] = {
        "description": "customer dataset format convert to COCO format",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2023,
        "contributor": "oysy",
        "date_created": "2023/08/17"
    }
    # coco annotations licenses.
    annotations["licenses"] = [{
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        "id": 1,
        "name": "Apache License 2.0"
    }]
    # coco annotations categories.
    annotations["categories"] = []
    for cls, clsname in enumerate(__CLASS__):
        if clsname == '__background__':
            continue
        annotations["categories"].append(
            {
                "supercategory": "car",
                "id": cls,
                "name": clsname
            }
        )
        #保存自己的标签
        for catdict in annotations["categories"]:
            if "car" == catdict["name"] and args.have_points:
                catdict["keypoints"] = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
                catdict["skeleton"] = [[1,2],[1,4],[2,3],[3,4],[5,6],[5,8],[6,7],[7,8],[9,10],[9,12],[10,11],[11,12],
                                       [13,14],[13,16],[14,15],[15,16]]

    annotations["images"] = []
    annotations["annotations"] = []
    #label文件夹下txt
    label_path = args.label_path
    print(label_path)
    #图片的绝对路径
    images_folder = phase_folder
    print("images_folder: ",images_folder)

    if os.path.isdir(label_path) and os.path.exists(images_folder):
        print("convert datasets {} to coco format!".format(images_folder))
        step = 0
        for id, line in enumerate(os.listdir(images_folder)):
            # print(line)
            #标签名字
            label_name = line.split('.')[0] + '.json'
            print("label_name: ",label_name)
            labeljson = os.path.join(label_path, label_name)
            fd = open(labeljson, "r")
            if fd:
                json_data = json.load(fd)
                points = json_data['shapes']
                landmarks = []
                for point in points:
                    for p in point['points']:
                        landmarks.append(p)

                # print(landmarks)
                landmarks = np.array(landmarks, dtype=np.int32)
                #图片的名字
                image_name = line

                if image_name:
                    # 找出x轴的最大最小值
                    x_min = np.min(landmarks[:, 0])
                    x_max = np.max(landmarks[:, 0])

                    # 找出y轴的最大最小值
                    y_min = np.min(landmarks[:, 1])
                    y_max = np.max(landmarks[:, 1])
                    bbox = [x_min,y_min, x_max, y_max]
                    cls = 1
                    x1 = x_min
                    y1 = y_min
                    bw = x_max - x_min
                    bh = y_max - y_min
                filename = os.path.join(images_folder, image_name)

                print(filename)
                #读取图片名字

                img = cv2.imread(filename)
                # print(img)
                # if args.have_check:
                #     for i in range(args.pointNum):
                #         cv2.circle(img,(int(float(label_info[5 + i*3])),int(float(label_info[6 + 3*i]))),2,(0,0,255),2)
                #     pthsave ="/mimages/" + str(id) +".jpg"
                #     cv2.imwrite(pthsave,img)
                # if img is None:
                #     continue
                height, width, _ = img.shape

                annotations["images"].append(
                    {
                        "license": 1,
                        "file_name": image_name,
                        "coco_url": "",
                        "height": height,
                        "width": width,
                        "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "flickr_url": "",
                        "id": id
                    }
                )
                # coco annotations annotations.
                annotations["annotations"].append(
                    {
                        "id": id,
                        "image_id": id,
                        "category_id": cls,
                        "segmentation": [[]],
                        "area": bw*bh,
                        # "bbox": [x1, y1, bw, bh],
                        "bbox": [0, 0, 1280, 704],
                        "iscrowd": 0,
                    }
                )
                #是否开启写入关键点
                if args.have_points:
                    catdict = annotations["annotations"][id]
                    coco_keypoints = []
                    for idx, [x, y] in enumerate(landmarks):
                        coco_keypoints.extend([x, y, 2])  # 添加x、y坐标和置信度
                    coco_keypoints = np.array(coco_keypoints).reshape(1, -1)
                        #points = [int(p) for p in label_info[2].split(",")]
                    catdict["keypoints"] = coco_keypoints.flatten().tolist()

                    catdict["num_keypoints"] = args.pointNum
                step += 1
                if step % 100 == 0:
                    print("processing {} ...".format(step))
        fd.close()

    else:
        print("WARNNING: file path incomplete, please check!")

    json_path = os.path.join(project_path, "train.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f,cls=JsonEncoder)


if __name__ == "__main__":
    print("begining to convert customer format to coco format!")
    args = argparser()
    try:
        MainProcessing(args)
    except Exception as e:
        traceback.print_exc()
    print("successful to convert customer format to coco format")
