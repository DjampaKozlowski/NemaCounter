# from __future__ import annotations

import tkinter as tk
import customtkinter as CTK
from PIL import Image as PILimage
from tkinter import filedialog, messagebox
import os
import numpy as np
import pandas as pd
import sys
import csv
import torch
import cv2
import threading
import nemacounter.utils as utils
import nemacounter.common as common
from nemacounter.detection import detection_workflow
from nemacounter.edition import edition_workflow
from nemacounter.segmentation import NemaCounterSegmentation, add_masks_on_image, create_multicolored_masks_image

CTK.set_appearance_mode("Dark")
CTK.set_default_color_theme("blue")


class NemaCounterGUI:

    def __init__(self):
        self.root = CTK.CTk()
        self.nb_wanted_cpu = CTK.IntVar()
        self.use_GPU = CTK.IntVar(value=1)
        self.scaling_var = CTK.StringVar(value="110%")
        self.theme_var = CTK.StringVar(value="Dark")
        self.tab_var = CTK.StringVar(value="Object Detection")
        self.yolo_model_var = CTK.StringVar()
        self.fpath_segany = ""
        self.set_main_window()

    def open_directory(self, var, label_obj):
        dpath = filedialog.askdirectory(initialdir='.', title='Select directory')
        if dpath:
            var.set(dpath)
            label_obj.configure(text=os.path.relpath(dpath))
        else:
            var.set('')

    def get_globinfo_fpath(self, var, label_obj):
        fpath = filedialog.askopenfilename(initialdir='.', title='Select a globinfo file',
                                           filetypes=[('csv files', '*_globinfo.csv')])
        if fpath:
            var.set(fpath)
            label_obj.configure(text=os.path.relpath(fpath))
        else:
            var.set('')

    def change_appearance_mode_event(self, _):
        CTK.set_appearance_mode(self.theme_var.get())

    def change_scaling_event(self, _):
        new_scaling_float = int(self.scaling_var.get().replace("%", "")) / 100
        CTK.set_widget_scaling(new_scaling_float)

    def set_main_window(self):
        self.root.title("Nemacounter by Thomas Baum lab")
        self.root.geometry(f"{1100}x{700}")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure((0, 1, 2), weight=1)
        self.set_side_bar()
        self.set_central_tabview()
        self.root.mainloop()

    def display_cpu_number(self, val):
        self.label_cpu_slider.configure(text=f"Max. number of CPU: {int(val)}")

    def display_confidence(self, val):
        self.label_conf_slider.configure(text=f"Confidence Threshold: {np.round(val, 2)}")

    def display_overlap(self, val):
        self.label_overl_slider.configure(text=f"Overlap Threshold: {np.round(val, 2)}")

    def set_side_bar(self):
        sidebar_frame = CTK.CTkFrame(master=self.root, width=140, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")

        logo_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, rowspan=4)
        nemacounter_logo = CTK.CTkImage(PILimage.open(os.path.relpath("conf/logo.png")), size=(300, 300))
        image_label = CTK.CTkLabel(master=logo_frame, image=nemacounter_logo, text='')
        image_label.grid(row=0, column=0)

        hardware_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        hardware_frame.grid(row=5, column=0, rowspan=2)
        switch_GPU = CTK.CTkSwitch(master=hardware_frame, variable=self.use_GPU,
                                   onvalue=1, offvalue=0, text=f"Use GPU if available")
        switch_GPU.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        nb_avail_cpu = utils.compute_available_cpu()
        self.label_cpu_slider = CTK.CTkLabel(master=hardware_frame, text=f"Max. number of CPU: {nb_avail_cpu - 1}",
                                             anchor="w")
        self.label_cpu_slider.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        slider_cpu = CTK.CTkSlider(master=hardware_frame, from_=1, to=nb_avail_cpu,
                                   number_of_steps=nb_avail_cpu, variable=self.nb_wanted_cpu,
                                   command=self.display_cpu_number)
        slider_cpu.set(nb_avail_cpu - 1)
        slider_cpu.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        displparams_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        displparams_frame.grid(row=7, column=0, rowspan=2)
        appearance_mode_label = CTK.CTkLabel(master=displparams_frame, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        appearance_mode_optionemenu = CTK.CTkOptionMenu(master=displparams_frame,
                                                        values=["Dark", "Light", "System"],
                                                        command=self.change_appearance_mode_event,
                                                        variable=self.theme_var)
        appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))

        scaling_label = CTK.CTkLabel(master=displparams_frame, text="UI Scaling:", anchor="w")
        scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        scaling_optionemenu = CTK.CTkOptionMenu(master=displparams_frame,
                                                values=["80%", "90%", "100%", "110%", "120%"],
                                                command=self.change_scaling_event,
                                                variable=self.scaling_var)
        scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

        modelspath_frame = CTK.CTkFrame(master=sidebar_frame, fg_color="transparent")
        modelspath_frame.grid(row=9, column=0, rowspan=2)

        yolo_label = CTK.CTkLabel(master=modelspath_frame, text="Select YOLO Model:", anchor="w")
        yolo_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky='w')

        yolo_files = [f for f in os.listdir("models") if f.endswith('.pt')]
        if not yolo_files:
            message = "No YOLO model files found in the models directory."
            tk.messagebox.showwarning("Warning", message)
            sys.exit()

        self.yolo_model_var.set(yolo_files[0])  # Select the first model by default
        yolo_model_menu = CTK.CTkOptionMenu(master=modelspath_frame, values=yolo_files, variable=self.yolo_model_var)
        yolo_model_menu.grid(row=1, column=0, padx=20, pady=(10, 10))

        fpath_conf = 'conf/config.ini'
        if os.path.exists(fpath_conf):
            config = common.get_config_info(os.path.abspath(fpath_conf))
            self.fpath_segany = os.path.relpath(config['Models']['seganymodel_path'])
            if not os.path.exists(self.fpath_segany):
                message = "Missing Segmentation model file. Please add the model path to the conf/config.ini file"
                tk.messagebox.showwarning("Warning", message)
                sys.exit()
        else:
            message = "No configuration file found"
            tk.messagebox.showwarning("Warning", message)
            sys.exit()

    def start_detection(self, indir_var, outdir_var, projid_entry, confslid_var, overslid_var, add_overlay_var):
        if not self.yolo_model_var.get():
            messagebox.showwarning("Warning", "You must select a YOLO model before launching the analysis.")
            return

        dct_var_detection = {
            'input_directory': indir_var.get(),
            'output_directory': outdir_var.get(),
            'project_id': projid_entry.get(),
            'conf_thresh': confslid_var.get(),
            'overlap_thresh': overslid_var.get(),
            'add_overlay': add_overlay_var.get(),
            'yolo_model': os.path.join("models", self.yolo_model_var.get()),
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get()}

        if (dct_var_detection['input_directory'] != '') and (dct_var_detection['output_directory'] != ''):
            project_outdir = os.path.join(dct_var_detection['output_directory'], dct_var_detection['project_id'])
            if os.path.exists(project_outdir):
                message = f"The folder '{project_outdir}' already exists."
                messagebox.showwarning("Warning", message)
            else:
                execution_log = detection_workflow(dct_var_detection, gui=True)
                if execution_log is None:
                    message = "Detection executed successfully."
                    messagebox.showinfo("Info", message)
                else:
                    message = "An error occurred during program execution:"
                    messagebox.showerror("Error", message)
        else:
            message = "You must select an input and an output directory before launching the analysis"
            messagebox.showwarning("Warning", message)

    def tabview_callback(self, value):
        selected_tab = value.get()
        if selected_tab == "Object Detection":
            self.set_detection_tab()
        elif selected_tab == "Manual Edition":
            self.set_edition_tab()
        elif selected_tab == "Object Segmentation":
            self.set_segmentation_tab()

    def set_central_tabview(self):
        tabview = CTK.CTkTabview(master=self.root,
                                 command=lambda: self.tabview_callback(tabview),
                                 fg_color="transparent")
        tabview.grid(row=0, column=1, rowspan=4,
                     padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.detection_tab = tabview.add("Object Detection")
        self.edition_tab = tabview.add("Manual Edition")
        self.segmentation_tab = tabview.add("Object Segmentation")

        if True:
            self.set_detection_tab()

    def set_detection_tab(self):
        projid_frame = CTK.CTkFrame(master=self.detection_tab,
                                    fg_color="transparent",
                                    width=500,
                                    height=50)
        projid_frame.grid(row=0, column=0, pady=(40, 10))
        projid_text = CTK.CTkLabel(master=projid_frame, text='Enter a project name')
        projid_text.grid(row=0, column=0, sticky='')
        projid_var = CTK.StringVar(value="MyProject")
        projid_entry = CTK.CTkEntry(master=projid_frame, placeholder_text=projid_var.get(),
                                    textvariable=projid_var)
        projid_entry.grid(row=1, column=0, sticky='')

        parameters_frame = CTK.CTkFrame(master=self.detection_tab,
                                        fg_color="transparent", width=500, height=400)
        parameters_frame.grid(row=1, column=0, pady=5)

        indir_frame = CTK.CTkFrame(master=parameters_frame,
                                   fg_color="transparent")
        indir_frame.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="w")
        indir_var = CTK.StringVar()
        indir_text = CTK.CTkLabel(master=indir_frame, text='Select Input Image Directory:')
        indir_text.grid(row=0, column=0, sticky="w")
        indir_button = CTK.CTkButton(master=indir_frame,
                                     text="Select",
                                     command=lambda: self.open_directory(indir_var, indir_label))
        indir_button.grid(row=1, column=0, padx=20)
        indir_label = CTK.CTkLabel(master=indir_frame, text='', anchor="w")
        indir_label.grid(row=1, column=2)

        outdir_frame = CTK.CTkFrame(master=parameters_frame,
                                    fg_color="transparent")
        outdir_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="w")
        outdir_var = CTK.StringVar()
        outdir_text = CTK.CTkLabel(master=outdir_frame, text='Select Output Directory:')
        outdir_text.grid(row=0, column=0, sticky="w")
        outdir_button = CTK.CTkButton(master=outdir_frame,
                                      text="Select",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.grid(row=1, column=0, padx=20)
        outdir_label = CTK.CTkLabel(master=outdir_frame, text='', anchor="w")
        outdir_label.grid(row=1, column=2)

        sliders_frame = CTK.CTkFrame(master=parameters_frame,
                                     fg_color="transparent")
        sliders_frame.grid(row=3, column=0, columnspan=4, padx=20, pady=20)
        confslid_var = CTK.DoubleVar()
        overslid_var = CTK.DoubleVar()
        default_conf_thresh = 0.5
        self.label_conf_slider = CTK.CTkLabel(master=sliders_frame,
                                              text=f"Confidence Threshold: {np.round(default_conf_thresh, 2)}",
                                              anchor="w")
        self.label_conf_slider.grid(row=0, column=0, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="ew")
        confidence_slider = CTK.CTkSlider(master=sliders_frame, from_=0, to=1,
                                          number_of_steps=100, variable=confslid_var,
                                          command=self.display_confidence)
        confidence_slider.grid(row=1, column=0, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="ew")
        confidence_slider.set(default_conf_thresh)

        default_overlap_val = 0.3
        self.label_overl_slider = CTK.CTkLabel(master=sliders_frame,
                                               text=f"Overlap Threshold: {np.round(default_overlap_val, 2)}",
                                               anchor="w")
        self.label_overl_slider.grid(row=0, column=2, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="w")
        overlap_slider = CTK.CTkSlider(master=sliders_frame, from_=0, to=1,
                                       number_of_steps=100, variable=overslid_var,
                                       command=self.display_overlap)
        overlap_slider.grid(row=1, column=2, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="w")
        overlap_slider.set(default_overlap_val)

        overlay_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        overlay_frame.grid(row=4, column=0, columnspan=4, padx=20, pady=20)
        overlay_switch_text = 'save images copies with the detection box overlay'
        add_overlay_var = CTK.IntVar(value=0)
        overlay_switch = CTK.CTkSwitch(master=overlay_frame, variable=add_overlay_var,
                                       onvalue=1, offvalue=0, text=overlay_switch_text)
        overlay_switch.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        start_button_frame = CTK.CTkFrame(master=self.detection_tab,
                                          fg_color="transparent", width=500, height=50)
        start_button_frame.grid(row=2, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Detection",
                                     command=lambda: self.start_detection(indir_var,
                                                                          outdir_var,
                                                                          projid_entry,
                                                                          confslid_var,
                                                                          overslid_var,
                                                                          add_overlay_var))
        start_button.grid(row=0, column=0)

        self.detection_tab.grid_columnconfigure(0, weight=1)


    def set_edition_tab(self):
        master_frame = CTK.CTkFrame(master=self.edition_tab, fg_color="transparent")
        master_frame.grid(row=0, column=0)
        projid_frame = CTK.CTkFrame(master=master_frame, fg_color="transparent")
        projid_frame.grid(row=0, column=0)
        projid_text = CTK.CTkLabel(master=projid_frame, text='Enter a project name')
        projid_text.grid(row=0, column=0, sticky='')
        projid_entry = CTK.CTkEntry(master=projid_frame, placeholder_text="MyProject")
        projid_entry.grid(row=1, column=0, sticky='')

        input_frame = CTK.CTkFrame(master=master_frame,
                                   fg_color="transparent",
                                   width=400, height=300)
        input_frame.grid(row=1, column=0, padx=20, pady=(20, 10), sticky='w')
        infile_var = CTK.StringVar()
        infile_text = CTK.CTkLabel(master=input_frame, text='Select *.globinfo.csv file:')
        infile_text.grid(row=0, column=0, sticky="w")
        infile_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.grid(row=1, column=0, padx=20)
        infile_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        infile_label.grid(row=1, column=2)

        outdir_var = CTK.StringVar()
        outdir_text = CTK.CTkLabel(master=input_frame, text='Select Output Directory:')
        outdir_text.grid(row=2, column=0, sticky="w")
        outdir_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.open_directory(outdir_var, outdir_label))
        outdir_button.grid(row=3, column=0, padx=20)
        outdir_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        outdir_label.grid(row=3, column=2)

        start_button_frame = CTK.CTkFrame(master=master_frame,
                                          fg_color="transparent", width=500, height=50)
        start_button_frame.grid(row=3, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Manual Edition",
                                     command=lambda: self.start_manual_edition(projid_entry, infile_var, outdir_var))
        start_button.grid(row=0, column=0, padx=175)

        self.edition_tab.grid_rowconfigure(0, weight=1)
        self.edition_tab.grid_columnconfigure(0, weight=1)

    def start_manual_edition(self, projid_entry, infile_var, outdir_var):
        fpath_globinfo = infile_var.get()
        dpath_parent = outdir_var.get()
        project_id = projid_entry.get()
        dpath_outdir = os.path.join(dpath_parent, project_id)

        if os.path.exists(fpath_globinfo) and dpath_outdir:
            if not os.path.exists(dpath_outdir):
                os.makedirs(dpath_outdir)
            try:
                edition_workflow(fpath_globinfo, dpath_outdir, project_id)
                messagebox.showinfo("Success", "Manual edition process completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            if not os.path.exists(fpath_globinfo):
                messagebox.showerror("Error", "Please provide an input *globinfo.csv file.")
            if not dpath_outdir:
                messagebox.showerror("Error", "Please provide an output directory.")

    def set_segmentation_tab(self):
        parameters_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                        fg_color="transparent", width=500, height=400)
        parameters_frame.grid(row=0, column=0, pady=5)

        input_frame = CTK.CTkFrame(master=parameters_frame,
                                   fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=20, pady=(20, 10))
        infile_var = CTK.StringVar()
        infile_text = CTK.CTkLabel(master=input_frame, text='Select *.globinfo.csv file:')
        infile_text.grid(row=0, column=0, sticky="w")
        infile_button = CTK.CTkButton(master=input_frame,
                                      text="Select",
                                      command=lambda: self.get_globinfo_fpath(infile_var, infile_label))
        infile_button.grid(row=1, column=0, padx=20)
        infile_label = CTK.CTkLabel(master=input_frame, text='', anchor="w")
        infile_label.grid(row=1, column=2)

        overlay_frame = CTK.CTkFrame(master=parameters_frame, fg_color="transparent")
        overlay_frame.grid(row=4, column=0, columnspan=4, padx=20, pady=20)
        overlay_switch_text = 'save images copies with the segmentation overlay'
        add_overlay_var = CTK.IntVar(value=0)
        overlay_switch = CTK.CTkSwitch(master=overlay_frame, variable=add_overlay_var,
                                       onvalue=1, offvalue=0, text=overlay_switch_text)
        overlay_switch.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        start_button_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                          fg_color="yellow", width=500, height=50)
        start_button_frame.grid(row=1, column=0, pady=10)

        start_button = CTK.CTkButton(master=start_button_frame,
                                     text="Start Segmentation",
                                     command=lambda: self.start_segmentation(infile_var, add_overlay_var))
        start_button.grid(row=0, column=0)

        self.progress_frame = CTK.CTkFrame(master=self.segmentation_tab,
                                           fg_color="transparent", width=500, height=50)
        self.progress_frame.grid(row=2, column=0, pady=10)
        self.progress_bar = CTK.CTkProgressBar(master=self.progress_frame)
        self.progress_bar.grid(row=0, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.progress_bar.set(0)

        # Add the processing label
        self.processing_label = CTK.CTkLabel(master=self.progress_frame, text="Processing...", anchor="w")
        self.processing_label.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.processing_label.grid_remove()  # Hide the label initially

        self.segmentation_tab.grid_columnconfigure(0, weight=1)

    def start_segmentation(self, infile_var, add_overlay_var):
        dct_var_segmentation = {
            'input_file': infile_var.get(),
            'add_overlay': add_overlay_var.get(),
            'segany': self.fpath_segany,
            'gpu': self.use_GPU.get(),
            'cpu': self.nb_wanted_cpu.get()}
        if dct_var_segmentation['input_file'] == '':
            message = f"No *_globinfo.csv file selected. Please select a file before launching the analysis."
            messagebox.showwarning("Warning", message)
        else:
            # Show processing label and run segmentation in a separate thread to keep UI responsive
            self.processing_label.grid()
            threading.Thread(target=self.run_segmentation_workflow, args=(dct_var_segmentation,)).start()

    def run_segmentation_workflow(self, dct_var_segmentation):
        dct_var_segmentation['project_id'] = os.path.basename(dct_var_segmentation['input_file']).replace('_globinfo.csv', '')
        dct_var_segmentation['input_dir'] = os.path.dirname(dct_var_segmentation['input_file'])

        gpu_if_avail = utils.get_bool(dct_var_segmentation['gpu'])
        add_overlay = utils.get_bool(dct_var_segmentation['add_overlay'])
        utils.set_cpu_usage(dct_var_segmentation['cpu'])
        device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_if_avail else 'cpu')

        if add_overlay:
            dpath_overlay = os.path.join(dct_var_segmentation['input_dir'], dct_var_segmentation['project_id'], 'img', 'segmentation')
            os.makedirs(dpath_overlay, exist_ok=True)

        if utils.check_file_existence(dct_var_segmentation['input_file']):
            df = pd.read_csv(dct_var_segmentation['input_file'])
            lst_img_paths = df['img_id'].unique()
            segmentation_model = NemaCounterSegmentation(dct_var_segmentation['segany'], device=device)
            df['surface'] = np.nan

            total_lines = len(df)
            completed_lines = 0

            for img_path in lst_img_paths:
                img = common.read_image(img_path)
                img_df = df[df['img_id'] == img_path]
                boxes = common.create_boxes(img_df)
                masks = segmentation_model.objects_segmentation(img, boxes)
                df.loc[df['img_id'] == img_path, 'surface'] = np.sum(np.sum(masks, axis=1), axis=1)

                if add_overlay:
                    add_masks_on_image(masks, img)
                    fpath_out_img = os.path.join(dpath_overlay, f"{dct_var_segmentation['project_id']}_{os.path.basename(img_path)}")
                    cv2.imwrite(fpath_out_img, img)

                    multicolored_img = create_multicolored_masks_image(masks)
                    fpath_out_multi = os.path.join(dpath_overlay,
                                                   f"{dct_var_segmentation['project_id']}_{os.path.basename(img_path)}_colored.png")
                    cv2.imwrite(fpath_out_multi, multicolored_img)

                completed_lines += len(img_df)
                progress = completed_lines / total_lines
                self.update_progress(progress)

            df.to_csv(dct_var_segmentation['input_file'], index=False)

            summary_fpath = os.path.join(dct_var_segmentation['input_dir'], f"{dct_var_segmentation['project_id']}_summary.csv")
            df_summary_original = pd.read_csv(summary_fpath)
            df_summary_new = common.create_summary_table(df, dct_var_segmentation['project_id'])
            df_summary = pd.merge(df_summary_original, df_summary_new[['img_id', 'surface_mean', 'surface_std']],
                                  on='img_id')
            df_summary.to_csv(summary_fpath, index=False)

            self.update_progress(1)  # Ensure progress bar is full
            self.processing_label.grid_remove()  # Hide processing label
            messagebox.showinfo("Success", "Object segmentation process completed successfully.")

    def update_progress(self, progress):
        self.progress_bar.set(progress)
        self.root.update_idletasks()


if __name__ == "__main__":
    app = NemaCounterGUI()
