import argparse
from pathlib import Path
from sys import platform
from model import lpr_model
from model import crnn_model
from models import *
from utils.datasets import *
from utils.utils import *
from PIL import Image

src_path ='./output/'


lpr = lpr_model.EAST_CRNN()
lpr.load(east_path="model/trained_model_weights/frozen_east_text_detection.pb",
         crnn_path="model/trained_model_weights/CRNN30.pth")

def get_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(src_path + "removed_noise.png", img)
    cv2.imwrite(src_path + "thres.png", img)
    result = lpr.predict('output/thres.png')
    return result


def detect(
        cfg,
        weights,
        images,
        output='output',
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.45,
        webcam=True,
        video=''
):
    device = torch_utils.select_device()
    model = Darknet(cfg, img_size)

    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location={'cuda:1':'cuda:0'})['model'])

    model.to(device).eval()

    if webcam:
        dataloader = LoadWebcam(img_size=img_size,video=video)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    cap = cv2.VideoCapture(str(video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(str(video.split('.')[0])+'.avi',fourcc, 15.0, (int(width), int(height)))

    cc = 0
    for i, (path, img, im0) in enumerate(dataloader):
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]

        if len(pred) > 0:
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                plate = im0[int(y1):int(y2),int(x1):int(x2)]
                cv2.imwrite('cropped/'+str(cc)+'.png',plate)
                cc=cc+1

                text = get_string(plate)
                text = text[0][0]
                # text = recogonize(plate)
                print(text)
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)],txt=text)
        
        ## Frame visualization
        out.write(im0)
        cv2.imshow("im",im0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/weight.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.10, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.10, help='iou threshold for non-maximum suppression')
    parser.add_argument('--video', type=str, default='./demo_video.mp4', help='your video file name')

    opt = parser.parse_args()
    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            video=opt.video
        )