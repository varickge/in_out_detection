import cv2
import sys
import torch
from pathlib import Path
from yoloDetect import *


from time import time
from tqdm import tqdm

# Parsing arguments for algorithm
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/door_model/weights/best.pt",
        help="model path or triton URL",
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--door-conf-thres",
        type=float,
        default=0.25,
        help="confidence threshold for door detection",
    )
    parser.add_argument(
        "--person-conf-thres",
        type=float,
        default=0.25,
        help="confidence threshold person detection",
    )

    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    opt = parser.parse_args()

    return opt


class InOutReID:
    def __init__(self):
        self.argparse = parse_opt()
        self.device = select_device(f"cuda:{self.argparse.device}")
        self.zone_thresh = 20
        self.cache_len = 5
        self.bbox_cache = np.zeros((self.cache_len, 4))
        self.in_out_cache = np.zeros(5)
        self.person_bbox = np.zeros((0, 4))
        self.out = cv2.VideoWriter('door_test.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1280, 720))
        

    def capture(self):
        global menu, menu_button
        
        # Initializing source
        if self.argparse.source == "0":
#             capture = cv2.VideoCapture("rtsp://Test_1:ararat@192.168.1.129:554/stream1")
            capture = cv2.VideoCapture("rtsp://Test_2:ararat@192.168.1.138:554/stream1")

        else:
            capture = cv2.VideoCapture(self.argparse.source)
        
        # Initializing Yolo object
        yolo = yoloDetect(
            device=self.device,
            iou_thres=self.argparse.iou_thres,
            max_det=self.argparse.max_det,
            agnostic=self.argparse.agnostic_nms,
            half=self.argparse.half,
            line_thickness=self.argparse.line_thickness,
        )
        
        # Initializing door/person Yolo
        door_yolo = DetectMultiBackend(
            weights=self.argparse.weights, device=self.device, fp16=self.argparse.half
        )
        person_yolo = DetectMultiBackend(
            weights="yolov5s.pt", device=self.device, fp16=self.argparse.half
        )

        door_bbox = np.zeros(0)
        
        # Infinite loop for frame capture and Yolo functional
        while True:
            ret, frame = capture.read()
            img = frame.copy()
            img = img[..., ::-1]
            
            # Saving button events in .txt files
            with open("buttonClick.txt", "r+") as f:
                str_ = f.readline()
                f.truncate(0)

            with open("multipleDetections.txt", "r+") as f2:
                str_2 = f2.readline()

            if str_2:
                # If multiple door detections, select a door
                for bbox_i in range(len(door_bbox)):
                    p1_door = (int(door_bbox[bbox_i][0]), int(door_bbox[bbox_i][1]))
                    p2_door = (int(door_bbox[bbox_i][2]), int(door_bbox[bbox_i][3]))
                    cv2.rectangle(
                        img[..., ::-1],
                        p1_door,
                        p2_door,
                        (255, 0, 0),
                        thickness=self.argparse.line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        img[..., ::-1],
                        f"Door {bbox_i + 1}",
                        (p1_door[0] - 10, p1_door[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
                if door_indx.get():
                    door_bbox = door_bbox[door_indx.get() - 1]
                    x, y, x_w, y_h = (
                        door_bbox[0] - self.zone_thresh,
                        door_bbox[1],
                        door_bbox[2] + self.zone_thresh,
                        door_bbox[3] + self.zone_thresh,
                    )
                    x, x_w, y, y_h = round_box(x, x_w, y, y_h, imgsz=img.shape[:2])
                    crop_size = (y, y_h, x, x_w)
                    imgsz = (y_h - y, x_w - x)
                    p1_door = (int(door_bbox[0]), int(door_bbox[1]))
                    p2_door = (int(door_bbox[2]), int(door_bbox[3]))
                    with open("multipleDetections.txt", "r+") as f2:
                        f2.truncate(0)
                else:
                    self.out.write(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1280, 720)))
                    yield img
                    continue

            if str_:
                door_indx = tk.IntVar()
                door_bbox = yolo(
                    door_yolo,
                    img,
                    img_size=(416, 416),
                    scale_with=(1080, 1920),
                    conf_thres=self.argparse.door_conf_thres,  # 720, 1280
                )[:, 1:]
                if not len(door_bbox):
                    print("No detections in frame")
                    continue
                elif len(door_bbox) > 1:
                    menu = tk.Menu(menu_button, tearoff=0)
                    menu.add_radiobutton(
                        label=f"No selected", value=0, variable=door_indx
                    )
                    for bbox_i in range(len(door_bbox)):
                        menu.add_radiobutton(
                            label=f"Door {bbox_i +1}",
                            value=bbox_i + 1,
                            variable=door_indx,
                        )
                    menu_button["menu"] = menu

                    with open("multipleDetections.txt", "w+") as f2:
                        f2.write("True")
                    continue
                else:
                    door_bbox = door_bbox[0]

                    x, y, x_w, y_h = (
                        door_bbox[0] - self.zone_thresh,
                        door_bbox[1],
                        door_bbox[2] + self.zone_thresh,
                        door_bbox[3] + self.zone_thresh,
                    )
                    x, x_w, y, y_h = round_box(x, x_w, y, y_h, imgsz=img.shape[:2])
                    crop_size = (y, y_h, x, x_w)
                    imgsz = (y_h - y, x_w - x)
                    p1_door = (int(door_bbox[0]), int(door_bbox[1]))
                    p2_door = (int(door_bbox[2]), int(door_bbox[3]))

            if len(door_bbox):
                # Engage person Yolo
                bboxes = yolo(
                    person_yolo,
                    img,
                    img_size=imgsz,
                    crop_size=crop_size,
                    scale_with=imgsz,
                    conf_thres=self.argparse.person_conf_thres
                ).astype(int)
                
                # Put door rectangles on frame 
                cv2.rectangle(
                    img[..., ::-1],
                    p1_door,
                    p2_door,
                    (0, 0, 255),
                    thickness=self.argparse.line_thickness,
                    lineType=cv2.LINE_AA,
                )
                person_bbox = bboxes[np.where(bboxes[:, 0] == 0)[0], 1:]
                self.bbox_cache[:-1] = self.bbox_cache[1:]
                self.in_out_cache[:-1] = self.in_out_cache[1:]

                if len(bboxes):
                    self.in_out_cache[-1] = 0
                    for bbox_person in person_bbox[:1]:
                        self.bbox_cache[-1] = bbox_person

                        # put rectangle to person
                        p1 = (int(bbox_person[0] + crop_size[2]), int(bbox_person[1] + crop_size[0]))
                        p2 = (int(bbox_person[2] + crop_size[2]), int(bbox_person[3] + crop_size[0]))
                        cv2.rectangle(
                            img[..., ::-1],
                            p1,
                            p2,
                            (255, 0, 0),
                            thickness=self.argparse.line_thickness,
                            lineType=cv2.LINE_AA,
                        )
                
                # If no person detections, check IoM and In/Out
                elif self.bbox_cache[-1].sum():
                    self.in_out_cache[-1] = 1
                    self.bbox_cache[-1, [0, 2]] = (
                        self.bbox_cache[-1, [0, 2]] + crop_size[2]
                    )
                    self.bbox_cache[-1, [1, 3]] = (
                        self.bbox_cache[-1, [1, 3]] + crop_size[0]
                    )
                    iom = box_iom(
                        torch.tensor([self.bbox_cache[-1]]), torch.tensor([door_bbox])
                    )
                    self.bbox_cache[-1] = np.zeros(4)

                    if iom > 0.8:
                        texti = "PERSON OUT"
                    else:
                        texti = "PERSON IN"
                else:
                    self.in_out_cache[-1] = 0

                if self.in_out_cache.sum():
                    color = (0, 0, 255)
                    if texti == "PERSON IN":
                        color = (0, 255, 0)
                    cv2.putText(
                        img[..., ::-1],
                        texti,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        color,
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
            self.out.write(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1280, 720)))
            yield img

    def main(self):
        global frame_generator, menu_button, menu
        frame_generator = self.capture()

        with open("buttonClick.txt", "w+") as f:
            f.write("")
        with open("multipleDetections.txt", "w+") as f2:
            f2.write("")

        global win
        win = tk.Tk()
        win.geometry("4000x3000")
        win.title("In/Out Detection")
        # Create a Label to capture the Video frames
        label = Label(win)
        label.pack(side="left")

        def callback():
            with open("buttonClick.txt", "r+") as f:
                f.write("True")

        def show_frames():
            global frame_generator

            img = next(frame_generator)

            img = cv2.resize(img, (1280, 720))

            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(1, show_frames)
        
        def close_save():
            self.out.release()
            win.destroy()

        button1 = tk.Button(win, text="Start", width=50, bd=7, command=show_frames)
        button1.pack(side="top")

        button2 = tk.Button(win, text="Detect Door", width=50, bd=7, command=callback)
        button2.pack(side="top")

        menu_button = tk.Menubutton(win, text="Select main door", width=50, bd=7)
        menu_button.pack(side="top")

        button3 = tk.Button(
            win, text="Close", width=50, bd=7, command=close_save
        )
        button3.pack(side="top")

        win.mainloop()


if __name__ == "__main__":
    InOutReID().main()
