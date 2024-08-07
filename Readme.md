# Nemacounter

## Overview
Nemacounter is a versatile tool designed for object detection, manual edition, and object segmentation using sequentially YOLOv5 Object Detection, OpenCV library and Segment Anything Model. This guide provides step-by-step instructions on using the software through its three main tabs: Object Detection, Manual Edition, and Object Segmentation (see below for a detailed user guide).


## Installation & Quick test

### Windows

1. **Download and Install Python for Windows:**
   - If you don't have Python installed, download and install it from [Python's official website](https://www.python.org/downloads/windows/) --> Window installer (x64 or x32).
	During installation do not forget to "add python to path".

2. **Download and Install Anaconda for Windows:**
   - If you don't have Anaconda installed, download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).

3. **Clone the Nemacounter folder from Github:**
   - Go to https://github.com/DjampaKozlowski/NemaCounter to clone/download the Nemacounter folder, and extract it on your Desktop.
   - Download the “sam_vit_h_4b8939.pth” model from : https://iastate.box.com/s/akpql0jlvbd5mmw26e2lgya4ul9h1xua 
     and the   “cystmodel.pt” from : https://iastate.box.com/s/e9kfpkjkrfpye0wgje205023xpcmcgvs
   - Drag and drop the two downloaded models into the “models” folder located into the Nemacounter folder cloned from Github.


4. **Create and Activate the Conda Environment:**
   - Open Anaconda Prompt (by searching "anaconda" in your windows search bar).
   - Once the terminal is open, create a new environment called "Nemacounter" by copy/pasting the following command and press Y when   proposed in the Anaconda command window:

            conda create -n Nemacounter python=3.10
      
   - Activate the environment by copy/pasting the following command:

            conda activate Nemacounter
     

   - Change directory to the Nemacounter folder in Anaconda command window by copy/pasting the following command:

         cd C:\Users\username\Desktop\Nemacounter

   #Make sure to replace username by your computer username in the path. If you extracted/cloned the Nemacounter folder elsewhere than your Desktop, you will have to modify the path accordingly. 

   - Install all the required dependencies by copy/pasting the following command:
 
          pip install -r requirements_windows.txt
     

    (Or "requirements_linux.txt" if using Linux)
 
   - Wait for all dependencies to be downloaded and installed in the Anaconda command window.

Now that Nemacounter is installed through Anaconda follow the steps below to run the software each time you need to open it:
   - Open “Anaconda Prompt” (by searching "anaconda" in your Windows search bar and click on it).
   - Activate the environment by copy/pasting the following command:

         conda activate Nemacounter
 
   - Change directory to the Nemacounter folder in Anaconda command window by copy/pasting the following command:

         cd C:\Users\username\Desktop\Nemacounter

   - Run the software in Anaconda command window by copy/pasting the following command:

         python Nemacounter_gui.py 


## User Guide 

### GUI

#### Left Sidebar Settings (Applies to All Tabs)
##### Use GPU if Available
- *Description*: Toggle the use of GPU for processing if available.
- *How to Use*: Click the switch to enable or disable GPU usage.
##### Max. Number of CPU
- *Description*: Set the maximum number of CPU cores for processing.
- *How to Use*: Drag the slider to adjust the number of CPU cores. The current number of selected cores is displayed above the slider.
##### Appearance Mode
- *Description*: Change the appearance mode of the software.
- *How to Use*: Click the dropdown menu and select the desired appearance mode (Dark, Light, or System).
##### UI Scaling
- *Description*: Adjust the scaling of the user interface.
- *How to Use*: Click the dropdown menu and select the desired scaling percentage.
##### Select YOLO Model
- *Description*: Choose the YOLO model (.pt file) for object detection.
- *How to Use*: Click the dropdown menu and select the desired YOLO model file from the list.
- *Note*: If a new model.pt is generated by a user, it can be simply added in the “models” folder inside of the “Nemacounter” folder to be selectable in the dropdown menu.

#### Object Detection Tab

##### Enter a Project Name
*Description*: Set the name of your project.
*How to Use*: Click the text box and type the desired project name.
##### Select Input Image Directory
*Description*: Choose the directory containing the images to be processed.
*How to Use*: Click the "Select" button and navigate to the desired directory. Click "OK" to confirm.
##### Select Output Directory
*Description*: Choose the directory where the output results will be saved.
*How to Use*: Click the "Select" button and navigate to the desired directory. Click "OK" to confirm.
##### Confidence Threshold
*Description*: Set the confidence threshold for object detection.
*How to Use*: Drag the slider to adjust the threshold. The current value is displayed above the slider.

##### Overlap Threshold
*Description*: Set the overlap threshold for object detection.
*How to Use*: Drag the slider to adjust the threshold. The current value is displayed above the slider.
##### Save Images with Detection Box Overlay
*Description*: Enable or disable saving images with detection boxes overlayed.
*How to Use*: Click the switch to toggle this option on or off.
##### Start Detection
*Description*: Begin the object detection process with the specified settings.
*How to Use*: Click the "Start Detection" button to start processing.


#### Manual Edition Tab
##### Enter a Project Name
*Description*: Set the name of your project.
*How to Use*: Click the text box and type the desired project name.


##### Select *.globinfo.csv File
*Description*: Choose the CSV file containing the globinfo data generated by the Object Detection tab.
*How to Use*: Click the "Select" button and navigate to the desired CSV file. Click "OK" to confirm.
##### Select Output Directory
*Description*: Choose the directory where the edited results will be saved.
*How to Use*: Click the "Select" button and navigate to the desired directory. Click "OK" to confirm.
##### Start Manual Edition
*Description*: Begin the manual edition process with the specified settings.
*How to Use*: Click the "Start Manual Edition" button to start processing.



#### Object Segmentation Tab
##### Select *.globinfo.csv File
*Description*: Choose the CSV file containing the globinfo data generated by the Object detection tab or the Manual Edition tab.
*How to Use*: Click the "Select" button and navigate to the desired CSV file. Click "OK" to confirm.
##### Save Images with Segmentation Overlay
*Description*: Enable or disable saving images with segmentation overlays.
*How to Use*: Click the switch to toggle this option on or off.
##### Start Segmentation
*Description*: Begin the object segmentation process with the specified settings.
*How to Use*: Click the "Start Segmentation" button to start processing.
##### Progress Bar
*Description*: Displays the progress of the segmentation process.
*How to Use*: Monitor the progress bar to see the current progress of the segmentation.