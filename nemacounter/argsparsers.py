import argparse

def detection_argument_parser():
    parser = argparse.ArgumentParser(prog="NemaCounter", description="Detect and segment nematodes from images")
    parser.add_argument("-i", "--input_directory", type=str, help="path to the images directory", required=True)
    parser.add_argument("-o", "--output_directory", type=str, help="path to the output directory. Will store several sub-directories. If the directory path already exist, the program will stop", required=True)
    parser.add_argument("-p", "--project_id", type=str, help="a session identifier. Will be added to the output file. Usefull to differenciate several instances.", default='my_session')
    parser.add_argument("-c", "--conf_thresh", type=float, help="Confidence threshold to apply. Only the detected object with a probability > to this value will be classified as nematodes", default=0.5)
    parser.add_argument("--add_overlay", type=int, help="Save a copy of the input image with the bouding boxes of the detected objects (default : 1 e.g. yes)", default=1, choices=[0,1])
    parser.add_argument("--yolo_model", type=str, help="path to the trained yolov5 model", default='models/bestv6.pt')
    parser.add_argument("--cpu", type=int, help="Number of usable CPUs (default -1 e.g. all)", default=-1)
    parser.add_argument("--gpu", type=int, help="If set to 1 (default) and a GPU supporting cuda is available, will use it", default=1, choices=[0,1])
    parser.add_argument("--overlap_thresh", type=float, help="Object detection max. overlap", default=0.3)
    return parser.parse_args()    



def edition_argument_parser():
    parser = argparse.ArgumentParser(prog="NemaCounter", description="Detect and segment nematodes from images")
    parser.add_argument("-i", "--input_file", type=str, help="path to the project directory", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="path to the output directory. Will store several sub-directories. If the directory path already exist, the program will stop", required=True)
    parser.add_argument("-p", "--project_id", type=str, help="a session identifier. Will be added to the output file. Usefull to differenciate several instances.", default='my_session')  
    return parser.parse_args()    


def segmentation_argument_parser():
    parser = argparse.ArgumentParser(prog="NemaCounter", description="Detect and segment nematodes from images")
    parser.add_argument("-i", "--input_file", type=str, help="path to the input file", required=True)
    # parser.add_argument("-p", "--project_id", type=str, help="a session identifier. Will be added to the output file. Usefull to differenciate several instances.", default='my_session')  
    # parser.add_argument("--segany_path", type=str, help="path to the trained segment anything model", default='models/sam_vit_h_4b8939.pth')

    # parser.add_argument("-o", "--output_directory", type=str, help="path to the output directory. Will store several sub-directories. If the directory path already exist, the program will stop", required=True)
    # parser.add_argument("-s", "--session_id", type=str, help="a session identifier. Will be added to the output file. Usefull to differenciate several instances.", default='my_session')
    # parser.add_argument("-c", "--conf_thresh", type=float, help="Confidence threshold to apply. Only the detected object with a probability > to this value will be classified as nematodes", default=0.8)
    # parser.add_argument("--add_boxes_overlay", type=int, help="Save a copy of the input image with the bouding boxes of the detected objects (default : 1 e.g. yes)", default=1, choices=[0,1])
    # parser.add_argument("--yolo_model", type=str, help="path to the trained yolov5 model", default='models/bestv6.pt')
    # #parser.add_argument("--cpu", type=int, help="Number of usable CPUs (default -1 e.g. all)", default=-1)
    # #parser.add_argument("--gpu", type=int, help="If set to 1 (default) and a GPU supporting cuda is available, will use it", default=1, choices=[0,1])
    # parser.add_argument("--overlap_thresh", type=float, help="Object detection max. overlap", default=0.3)
    return parser.parse_args()  


# TODO : 
# - rework the descriptions
# - check the argsparse have every info the programs needs.