
import cv2
import pandas as pd
import numpy as np
import os

import nemacounter.utils as utils 
import nemacounter.common as common


class BoxEditor:
    
    def __init__(self, img_path, boxes):
        """
        Requires the image path and the list of box 
        [[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ymax], ...]
        """
        self.img_path = img_path
        self.boxes = boxes
        self.drawing = False
        self.selected_box = -1
        self.new_boxes = []
        self.deleted_boxes = []
        self.img = None 
        self.img_original = None
        
    def edit_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.new_boxes.append([x, y, x, y])

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.new_boxes[-1][2], self.new_boxes[-1][3] = x, y
            else:
                self.highlight_box(x, y)
            self.redraw_image()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.new_boxes[-1][2], self.new_boxes[-1][3] = x, y
                
    def highlight_box(self, x, y):
        self.selected_box = -1
        for i, box in enumerate(self.boxes + self.new_boxes):
            if box[0] < x < box[2] and box[1] < y < box[3]:
                self.selected_box = i
                
    def redraw_image(self):
        if self.img_original is not None:
            self.img = self.img_original.copy()
            self.draw_legend_on_image()
            for i, box in enumerate(self.boxes + self.new_boxes):
                if self.selected_box != -1 and i == self.selected_box:
                    color = (255, 0, 0)  # Color for the selected box
                else:
                    color = (0, 0, 255)  # Color for other boxes
                cv2.rectangle(self.img, (box[0], box[1]), (box[2], box[3]), color, 2)

                
    def draw_legend_on_image(self):
        instructions = ["Left Mouse Button: Draw Box",
                        "R Key: Remove Hovering Box",
                        "S Key: Save & Exit Annotations",
                        "ESC Key: Exit Without Saving Annotation"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 1
        position = (10, 30)
        line_height = 15
        for i, line in enumerate(instructions):
            cv2.putText(self.img, line, (position[0], position[1] + i * line_height), 
                        font, font_scale, color, thickness, cv2.LINE_AA)
            
    def remove_selected_box(self):
        if self.selected_box != -1:
            if self.selected_box < len(self.boxes):
                # Keep track of the deleted boxes from the original set
                self.deleted_boxes.append(self.boxes[self.selected_box])
                # Remove from pre-detected boxes
                del self.boxes[self.selected_box]
            else:
                # Remove from newly drawn boxes
                index_new_box = self.selected_box - len(self.boxes)
                del self.new_boxes[index_new_box]
            self.selected_box = -1  # Reset selected_box after removal
            self.redraw_image()
            
    def run(self):
        """
        Edition workflow
        
        Read the image and allow a variety of actions
        
        If the user press ESC key : exit without saving. return None
        If the user press s key : save and exit. return the potentially 
        modified input list of boxes as well as the potential new boxes 
        
        """
        # self.img = cv2.imread(self.img_path)
        # self.img_original = self.img.copy()
        self.img_original = cv2.imread(os.path.relpath(self.img_path))
        self.redraw_image()
        cv2.namedWindow(f'Annotation Tool - {os.path.relpath(self.img_path)}')
        cv2.setMouseCallback(f'Annotation Tool - {os.path.relpath(self.img_path)}',
                             self.edit_box)
        tag = None
        while True:
            cv2.imshow(f'Annotation Tool - {os.path.relpath(self.img_path)}', self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC key to exit
                tag = 'exit'
                break
            elif k == ord('s'):  # Save on pressing 's'
                tag = 'save'
                break                
            elif k == ord('r'):  # Remove selected box on pressing 'r'
                self.remove_selected_box()

        cv2.destroyAllWindows()
        if tag == 'exit':
            return None
        elif tag == 'save':
            return self.new_boxes, self.deleted_boxes     
        
def build_boxes_dataframe(boxes, img_path):
    df = pd.DataFrame(boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax'])
    df['img_id'] = img_path
    df = df[['img_id', 'xmin', 'ymin', 'xmax', 'ymax']]
    return df       


def edition_workflow(dct_args):    
    # Check that we can find the input *_globinfos.tsv file
    #globinfo_fpath = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_nemacounter_globinfo.tsv")
    globinfo_fpath = dct_args['input_file']
    
    #
    #   TODO : check that the ouput_dir/project_id is different from the original one
    #
    
    if utils.check_file_existence(globinfo_fpath):
        # Read the input *_globinfos.tsv file were all the information is stored
        df_input = pd.read_csv(globinfo_fpath, sep='\t')     
        
        # Delete the surface column if present (i.e. the user already used the segmentation tool.)
        # The user will have to re-run the segmentation analysis. Or maybe we could onle perform the s
        # segmentation for lines with NA to speed up the process

        # List the images in the input file
        lst_img_paths = df_input['img_id'].unique()
        
        #####
        # TMP
        #
        #
        df_input = df_input[df_input.img_id.isin(lst_img_paths[0:2])]
        #lst_img_paths = lst_img_paths[0:2]
        #
        #
        #####
        
        lst_df_new = []
        lst_df_deleted = []

        # Iterate through images slicing the input dataframe per image
        for img_path in lst_img_paths:
            # get the object bounding boxes coordinates
            original_boxes = common.create_boxes(df_input[df_input['img_id']==img_path])
            box_editor = BoxEditor(img_path, original_boxes)
            new_boxes, deleted_boxes = box_editor.run()
            lst_df_deleted.append(build_boxes_dataframe(deleted_boxes, img_path))
            lst_df_new.append(build_boxes_dataframe(new_boxes, img_path))
        df_new_boxes = pd.concat(lst_df_new)
        df_deleted_boxes = pd.concat(lst_df_deleted)
        
        # TODO : manage the exception when exiting annotation
        
        # Exclude the deleted boxes from the inpu dataframe
        tmp_df = df_input.drop(columns=['project_id', 'object_id'])\
            .merge(df_deleted_boxes, how='left', indicator=True)
        df_global = tmp_df[tmp_df['_merge'] == 'left_only'].drop(columns='_merge')
        
        ## Add the new boxes 
        # Align column structures by adding missing columns with NaN values
        for col in df_global.columns:
            if col not in df_new_boxes.columns:
                df_new_boxes[col] = np.nan
                
        df_global = pd.concat([df_global, df_new_boxes],ignore_index=True)
        
        # Insert the project_id as a first column
        df_global.insert(loc=0, column='project_id', value=dct_args['project_id'])
        
        # Insert the object_id as a third column
        df_global.insert(loc=2, column='object_id', value=df_global.groupby('img_id').cumcount()+1)
        
        # Save the global dataframe
        df_global.to_csv(os.path.join(dct_args['output_directory'], f"{dct_args['project_id']}_manualedition_globinfo.tsv"), 
                                    sep='\t', index=False)
    
        
        ##  Create a summary (per image) table.
        df_summary = common.create_summary_table(df_global, dct_args['project_id'])
        
        ## Count the number of deleted and added boxes per image
        df_cnt_new = df_new_boxes.reset_index().groupby('img_id')[['img_id']].size().reset_index(name='new_detection')
        df_cnt_deleted = df_deleted_boxes.reset_index().groupby('img_id')[['img_id']].size().reset_index(name='deleted_detection')
        
        ## Add the info to the summary dataframe
        df_summary = df_summary.merge(df_cnt_new, how='left', on='img_id')
        df_summary = df_summary.merge(df_cnt_deleted, how='left', on='img_id')
        df_summary.to_csv(os.path.join(dct_args['output_directory'], f"{dct_args['project_id']}_manualedition_summary.tsv"), 
                        sep='\t', index=False)
    else:
        print("File not found")
