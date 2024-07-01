import cv2
import pandas as pd
import numpy as np
import os

import nemacounter.common as common

# Initialize global variables
drawing = False
selected_box = -1
new_boxes = []
deleted_boxes = []
img = None
img_original = None
boxes = []

def edit_box(event, x, y, flags, param):
    global drawing, new_boxes, img, selected_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        new_boxes.append([x, y, x, y])
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            new_boxes[-1][2], new_boxes[-1][3] = x, y
        else:
            highlight_box(x, y)
        redraw_image()
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            # Normalize the box to ensure xmin < xmax and ymin < ymax
            x_min, x_max = sorted([new_boxes[-1][0], x])
            y_min, y_max = sorted([new_boxes[-1][1], y])
            new_boxes[-1] = [x_min, y_min, x_max, y_max]
            redraw_image()

def highlight_box(x, y):
    global selected_box, boxes, new_boxes
    selected_box = -1
    for i, box in enumerate(boxes + new_boxes):
        x_min = min(box[0], box[2])
        x_max = max(box[0], box[2])
        y_min = min(box[1], box[3])
        y_max = max(box[1], box[3])
        if x_min <= x <= x_max and y_min <= y <= y_max:
            selected_box = i

def redraw_image():
    global img, img_original, boxes, new_boxes, selected_box
    if img_original is not None:
        img = img_original.copy()
        for i, box in enumerate(boxes + new_boxes):
            color = (255, 0, 0) if i == selected_box else (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        draw_legend_on_image()

def draw_legend_on_image():
    global img
    instructions = ["Left Mouse Button: Draw Box", "R Key: Remove Hovering Box", "S Key: Save & Exit Annotations",
                    "ESC Key: Exit Without Saving Annotation"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.5
    font_scale = 1

    color = (255, 255, 255)
    position = (10, 30)
    for i, line in enumerate(instructions):
        cv2.putText(img, line, (position[0], position[1] + 15 * i), font, font_scale, color, 1, cv2.LINE_AA)

def remove_selected_box():
    global boxes, new_boxes, selected_box, deleted_boxes
    if selected_box != -1:
        if selected_box < len(boxes):
            deleted_boxes.append(boxes.pop(selected_box))
        else:
            deleted_boxes.append(new_boxes.pop(selected_box - len(boxes)))
        selected_box = -1
        redraw_image()

def run_manual_annotation(img_path, box_data):
    global img, img_original, boxes, new_boxes, deleted_boxes
    img = cv2.imread(img_path)
    cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
    img_original = img.copy()
    boxes = box_data.copy()
    cv2.setMouseCallback('Annotation Tool', edit_box)
    cv2.imshow('Annotation Tool', img)

    dct_res = {}
    while True:
        cv2.imshow('Annotation Tool', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            new_boxes = []
            deleted_boxes = []
            break
        elif k == ord('s'):  # Save on pressing 's'
            dct_res = {'img_id': img_path,
                       'new_boxes': new_boxes.copy(),
                       'actual_boxes': boxes.copy(),
                       'deleted_boxes': deleted_boxes.copy()}
            new_boxes = []
            deleted_boxes = []
            break
        elif k == ord('r'):  # Remove selected box on pressing 'r'
            remove_selected_box()

    cv2.destroyAllWindows()
    return dct_res

def build_boxes_dataframe(boxes, img_path):
    df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    df['img_id'] = img_path
    return df[['img_id', 'xmin', 'ymin', 'xmax', 'ymax']]


def edition_workflow(input_file, output_directory, project_id):
    if os.path.exists(input_file):
        df_input = pd.read_csv(input_file)
        lst_img_paths = df_input['img_id'].unique()
        lst_df_actual = []
        lst_df_new = []
        lst_df_deleted = []
        for img_path in lst_img_paths:
            full_img_path = os.path.join(os.path.dirname(input_file), img_path)
            box_data = common.create_boxes(df_input[df_input['img_id'] == img_path])
            dct_res = run_manual_annotation(full_img_path, box_data)
            # Case when boxes added and / or removed
            if len(dct_res) > 0:
                if len(dct_res['new_boxes']) > 0:
                    lst_df_new.append(build_boxes_dataframe(dct_res['new_boxes'], full_img_path))
                if len(dct_res['actual_boxes']) > 0:
                    lst_df_actual.append(build_boxes_dataframe(dct_res['actual_boxes'], full_img_path))
                if len(dct_res['deleted_boxes']) > 0:
                    lst_df_deleted.append(build_boxes_dataframe(dct_res['deleted_boxes'], full_img_path))

            # Case where nothing happened : output original boxes
            else:
                lst_df_actual.append(build_boxes_dataframe(box_data, full_img_path))

        if len(lst_df_actual) > 0:
            df_actual_boxes = pd.concat(lst_df_actual)
            df_actual_boxes = df_actual_boxes.merge(df_input[['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence']],
                                                    how='left',
                                                    on=['img_id', 'xmin', 'ymin', 'xmax', 'ymax'])
        else:
            df_actual_boxes = pd.DataFrame()

        if len(lst_df_new) > 0:
            df_new_boxes = pd.concat(lst_df_new)
            df_new_boxes['confidence'] = np.nan
        else:
            df_new_boxes = pd.DataFrame()

        if len(lst_df_deleted) > 0:
            df_deleted_boxes = pd.concat(lst_df_deleted)
        else:
            df_deleted_boxes = pd.DataFrame()

        df_global = pd.concat([df_actual_boxes, df_new_boxes], ignore_index=True)
        # Insert the project_id as a first column
        df_global.insert(loc=0, column='project_id', value=project_id)
        # Insert the object_id as a third column
        df_global.insert(loc=2, column='object_id', value=df_global.groupby('img_id').cumcount() + 1)
        df_global.to_csv(os.path.join(output_directory, f"{project_id}_manualedition_globinfo.csv"), index=False)

        if df_global.shape[0] != 0:
            ##  Create a summary (per image) table.
            df_summary = common.create_summary_table(df_global, project_id)

            ## Count the number of deleted and added boxes per image
            if len(df_new_boxes) != 0:
                df_cnt_new = df_new_boxes.reset_index().groupby('img_id')[['img_id']].size().reset_index(
                    name='new_detection')
                df_summary = df_summary.merge(df_cnt_new, how='left', on='img_id')
            else:
                df_summary['new_detection'] = np.nan

            if len(df_deleted_boxes) != 0:
                df_cnt_deleted = df_deleted_boxes.reset_index().groupby('img_id')[['img_id']].size().reset_index(
                    name='deleted_detection')
                df_summary = df_summary.merge(df_cnt_deleted, how='left', on='img_id')
            else:
                df_summary['deleted_detection'] = np.nan
        else:
            df_summary = pd.DataFrame(columns=df_input.columns.values)
        df_summary = df_summary.fillna(int(0))
        df_summary.to_csv(os.path.join(output_directory, f"{project_id}_manualedition_summary.csv"), index=False)

        print("Success", "Annotations saved successfully.")
    else:
        print("Error", "The specified file does not exist.")
