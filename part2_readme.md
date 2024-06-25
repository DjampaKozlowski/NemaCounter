

python : 3.10.14 






# Usage

Detection:
`python Nemacounter_detection.py -i data/toy -o tmp/ --add_boxes_overlay 0`


Segmentation:
`python Nemacounter_segmentation.py -i tmp/ -p my_session`


Edition:
`python Nemacounter_edition.py -i tmp/my_session_nemacounter_globinfo.tsv -p my_session2 -o tmp2/`


In case the user wants to use differents models, their files paths must be updated. Using the graphical user interface (GUI), the user only need to manually navigate to the directory where the model, select the model file and click ok. The models changes will be saved in a configuration file and will be persistent. Using the command line interface (CLI), the user need to update the models path in the `conf/config.ini` file. 



# Installation 

In case the user wants to perform a manual installation in a virtual environment, it is advised to prefer virtualenv to conda (anaconda, miniconda, miniforge) because conda create graphical bugs due to a missing library.  


- https://github.com/batonogov/docker-pyinstaller
- https://customtkinter.tomschimansky.com/documentation/packaging/















# Futur improvements:

using GPU induces little to no improvement of the processing speed in comparison to CPU usage. 
First guess whould be that trating one image at the time is the cause. 
Maybe add a GPU mode that load and process all the images at once ?  


Allow to add labels to identify each object on the images.



# BUGS / FIX:

"Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory" 
```
ldconfig -p | grep cuda
# libcuda.so.1 (libc6,x86-64) => /usr/lib/wsl/lib/libcuda.so.1
nano .bashrc
# add export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```








