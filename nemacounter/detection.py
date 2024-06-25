import yolov5
import torch
import os
import cv2
import nemacounter.utils as utils
import nemacounter.common as common
import sys

class NemaCounterDetection:

    def __init__(self, weights_path, conf_thesh=0.5, iou_thresh=0.3, device='cpu'):
        #Todo : add device to speed up the process ?
        ##  Detect if a GPU is available to speed up the process
        self.device = device

        self.custom_model = yolov5.load(weights_path)

        #   set the model parameters. Detections with a conf value inferior to conf_thresh will not be outputed
        self.custom_model.conf = conf_thesh
        self.custom_model.iou = iou_thresh
        self.input_size = 1040 # dimensions used for resizing before inference (e.g. size used for finetunning)

        # TODO :
        # - add a function that parse a list of images with the option to parallelize (https://github.com/ultralytics/yolov5/issues/2960)
        
    def detect_objects(self, img):
        """
        Identify nematodes from a picture

        Returns :
        ---------
        (pandas dataframe)
        (else ?)
        """
        #   Detect objects
        df = self.custom_model(img, self.input_size).pandas().xyxy[0]
        #   Format Coordinates
        df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
        return df
    


def add_boxes_on_image(boxes, img):
    ##  Iterate over each box coordinates
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        # draw a red rectangle on the image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)        
    
def create_project_dirs_structure(dpath_outdir, project_id, display_boxes=False):
    dpath_project = os.path.join(dpath_outdir, project_id)
    if os.path.isdir(dpath_project):
        print('Output directory already exists, program will stop.')
        sys.exit()
    else:
        os.makedirs(dpath_project, mode=0o755)
        if display_boxes:
            os.makedirs(os.path.join(dpath_project, 'img', 'bounding_boxes'), mode=0o755)
            
            
def detection_workflow(dct_args, gui=True):
    # try:
    #   Get the model file path
    config = common.get_config_info(os.path.relpath('conf/config.ini'))
    fpath_model = config['Models']['yolomodel_path']
    if os.path.exists(fpath_model):
        #   Setup the ressources
        gpu_if_avail = utils.get_bool(dct_args['gpu'])
        add_boxes_overlay = utils.get_bool(dct_args['add_overlay'])
        utils.set_cpu_usage(dct_args['cpu'])
        #   Detect if a GPU is available (and the user wants to use it) to speed up the process
        #device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_if_avail==True else 'cpu')
        device = 'cpu'

        #   List the input image file paths
        lst_img_paths = utils.list_image_files(dct_args['input_directory']) # TODO : check the authorized input files
        
        #   Create the output directory. If running through GUI, do not check if the dir exist
        create_project_dirs_structure(dct_args['output_directory'], dct_args['project_id'], 
                                            display_boxes=add_boxes_overlay)

                
        ##  Init the dection  
        detection_model = NemaCounterDetection(os.path.relpath(dct_args['yolo_model']), 
                                            conf_thesh=dct_args['conf_thresh'], 
                                            iou_thresh=dct_args['overlap_thresh'],
                                            device=device)

        ##  Iterate through images
        #
        #   init the list where per image df will be stored
        lst_df = []
        for img_path in lst_img_paths:
            #   read the image
            img = common.read_image(img_path)        
            #   perform the detection
            df = detection_model.detect_objects(img)
            #   extract the image name and store it in the df
            df['img_id'] = img_path
            #   give an identifier to the detected objects 
            df['object_id'] = df.index.values+1
            #   if the user wants to visualise the detection
            if add_boxes_overlay==True:
                #   compute the objects bounding boxes
                boxes = common.create_boxes(df)
                #   add the segmented objects on top of the original image (and potentially the segmentation overlay)
                add_boxes_on_image(boxes, img)
                #   save the image in INPUT_DIR/PROJECT_ID/img/bounding_boxes                               img)
                fpath_out_img = os.path.join(dct_args['output_directory'], 
                                            dct_args['project_id'],
                                            'img/bounding_boxes', 
                                            f"{dct_args['project_id']}_{os.path.basename(img_path)}")
                if not cv2.imwrite(fpath_out_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
                    raise Exception("Could not write image") 
            lst_df.append(df)  
            
            
        ##  Create the global dataframe that contains all the information and save id
        dpath_stats = os.path.join(dct_args['output_directory'], dct_args['project_id'])
        df_global = common.create_global_table(lst_df, dct_args['project_id'])
        df_global.to_csv(os.path.join(dpath_stats, 
                                    f"{dct_args['project_id']}_globinfo.csv"), 
                                    index=False)
        
        ##  Create a summary (per image) table.
        df_summary = common.create_summary_table(df_global, dct_args['project_id'])
        df_summary.to_csv(os.path.join(dpath_stats, 
                                    f"{dct_args['project_id']}_summary.csv"),
                                    index=False)
        return None
    else:
        return False
    # except:
    #     raise Exception()
    #     return False
    
    