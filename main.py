from my_yolov6 import my_yolov6
import cv2
import csv
import time

yolov6_model = my_yolov6("weights/yolov6m.pt","cpu","data/coco.yaml", 640, True)


# define a video capture object
vid = cv2.VideoCapture('/media/huy/c05b0f00-d3be-4144-aa0d-01820543f0eb/huy/TL-tech/dataset/verhical/videorun/cam 2/1809.mp4')
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

    # print('boxs: ', det)

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
    names = [ 'Direction','car', 'motorcycle','bus', 'truck']
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
