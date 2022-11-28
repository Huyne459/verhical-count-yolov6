from my_yolov6 import my_yolov6
import cv2
import csv
import time

yolov6_model = my_yolov6("weights/yolov6m.pt","cpu","data/coco.yaml", 640, True)


# define a video capture object
vid = cv2.VideoCapture('/home/huy/Desktop/TL-tech/Verhical_count/data/videorun/cam 2/1809.mp4')
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 10.0, size)
count = 0
while (True):
    start = time.time()
    middle_line_position = 700
    up_line_position = middle_line_position - 30
    down_line_position = middle_line_position + 30
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    print('count: ', count)
    count += 1
    try:
        ih, iw, channels = frame.shape
        frame, len_det, det, up_list, down_list, up_detail, down_detail = yolov6_model.infer(frame, conf_thres=0.4, iou_thres=0.45, frame_now=count)
    # print('up_list:', up_list)
    # print('down_list:', down_list)
    except:
        break

    print('boxs: ', det)

    # Display the resulting frame
    cv2.line(frame, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
    cv2.line(frame, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
    cv2.line(frame, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('frame', frame)

    end = time.time()
    print("FPS: ", 1/(end-start))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("data.csv", 'w') as f1:
    cwriter = csv.writer(f1)
    names = [ 'Direction','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    cwriter.writerow(names)
    up_list.insert(0, "Up")
    down_list.insert(0, "Down")
    cwriter.writerow(up_list)
    cwriter.writerow(down_list)
    f1.close()

# with open("data2.csv", 'w') as f2:
#     detail_writer = csv.writer(f2)
#
#     detail_writer.writerow("Up:")
#     for row in up_detail:
#         detail_writer.writerow(row)
#
#     detail_writer.writerow("Down:")
#     for row in down_detail:
#         detail_writer.writerow(row)

# After the loop release the cap object
vid.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()