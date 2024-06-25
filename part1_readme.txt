# Nemacounter Installation

## Installation Instructions
1. **Download and Install Python for Windows:**
   - If you don't have Python installed, download and install it from [Python's official website](https://www.python.org/downloads/windows/) --> Window installer (x64 or x32).
	During installation do not forget to "add python to path".

2. **Download and Install Anaconda for Windows:**
   - If you don't have Anaconda installed, download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).

3. **Clone the Repository:**
   - Clone this repository from Github or download the ZIP file and extract it to your desired location.
   OR
   **Download the ZIP file:**
     Download the ZIP file from XXXXXXXXXXXXXXXXXXXXXX, and extract it to your desired location on your computer.

4. **Create and Activate the Conda Environment:**
   - Open Anaconda Prompt (by searching "anaconda" in your windows search bar).
   - Once the terminal is open, create a new environment called "Nemacounter" by copy/pasting the following command and press Y when   proposed in the Anaconda command window:

     conda create -n Nemacounter python=3.10

   -Activate the environment by copy/pasting the following command:

     conda activate Nemacounter

   -Change directory to the Nemacounter folder in Anaconda command window by copy/pasting the following command:
  
     cd C:\Users\username\Desktop\Nemacounter

     *Replace "username" by your computer username


   -Install all the required dependencies by copy/pasting the following command:
 
     pip install -r requirements.txt
 
   -Wait for all dependencies to be downloaded and installed in the Anaconda command window.

