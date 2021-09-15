import subprocess
from stereo2PCL import calibration_store
from stereo2PCL import stereo_depth_video
import detect
import argparse
import os

#exec(open("detect.py").read())
#exec (open('stereo2PCL/stereo_depth_video.py').read())


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\video\\left_out.avi', help='file/dir/URL/glob, '
                                                                                                                         '0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def parse_opt_depth():
    parser= argparse.ArgumentParser()
    parser.add_argument('--calibration_file', type=str,
                        default="C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\stereo2PCL\\configs\\stereo.yml")
    parser.add_argument('--left_source', default= "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\video\\left_out.avi" )
    parser.add_argument('--right_source', default= "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\video\\right_out.avi" )
    parser.add_argument('--pointcloud_dir', type=str, default= "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\stereo2PCL\\output\\")

    opt2 = parser.parse_args()
    return opt2


def main(opt, opt2):
    #print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    #check_requirements(exclude=('tensorboard', 'thop'))
    x1, y1, x2, y2, confidence_score,class_index, object_name, dataset = detect.run(**vars(opt))

    print(object_name, confidence_score)
    """
        while opt2.left_source:
        if (object_name == "apple" and float(confidence_score) >= 0.65):
            if x1 >= 0.4 *opt.imgsz:
                os.system("python C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\stereo2PCL\\stereo_depth_video.py --calibration_file "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\stereo2PCL\\configs\\stereo.yml  --left_source "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\video\\left_out.avi  --right_source  "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\video\\right_out.avi  --pointcloud_dir  "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\stereo2PCL\\output\\")
            else:
                pass
        else:
            pass

    while opt2.right_source:
        if (object_name == "apple" and float(confidence_score) >= 0.65):
            if x2 <= opt.imgsz - (0.4 *opt.imgsz):
                os.system("python C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\stereo2PCL\\stereo_depth_video.py --calibration_file "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\stereo2PCL\\configs\\stereo.yml  --left_source "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\video\\left_out.avi  --right_source  "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\video\\right_out.avi  --pointcloud_dir  "
                          "C:\\Users\\Clark\\Documents\\GitHub\\yolov5\\openCVStereo\\stereo2PCL\\output\\")
            else:
                pass
        else:
            pass
    """





if __name__ == "__main__":
    opt = parse_opt()
    opt2 = parse_opt_depth()
    main(opt, opt2)
