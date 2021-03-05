# Interactive Lung Tumour Segmentation Software
An open source software for Interactive Lung Tumour segmentation based on **Matlab** GUI. This software provides a tool to integrate prior knowledge of clinical experts using a few seeds with data-driven algorithms based on Random Walker. The graphical interface provides view-mode and draw-mode for image viewing and seed drawing respectively. It has been used in the research project for survival analysis of local advanced non-small cell lung cancer using CT images. ![software interface](https://github.com/bowenxin/lung_seg_interactive/blob/main/images/interface.png)

# Functions
- **View-mode** (defaut)
  - *Load Panel*: For loading CT images.
  - *Status and Patients Panel*: For showing image resolution, patient ID.
  - *CT Window Panel*: 1) Lung window, 2) Mediastinum window
  - *Sliding Panel*: For browing images or locating a specific scan.
- **Draw-mode** (interactive segmentation)
  - *Control Panel*-draw: activate draw mode
  - *Control Panel*-clear: activate eraser to remove seeds
  - *Control Panel*-save: save drawed seeds
  - *Control Panel*-delete all: remove all seeds of current patients and start again

# Citation
Wang, Linlin, et al. "Integrative nomogram of CT imaging, clinical, and hematological features for survival prediction of patients with locally advanced non-small cell lung cancer." 
European radiology 29.6 (2019): 2958-2967.

# Video demonstration
Will be released soon.

# How to use
1) Install matlab.
2) Open softare interface.
    - Add LungRWApp directory into matlab path.
    - Run lungRWapp.m file to open GUI interface. 
3) Load images (*Load panel*).
    - Input or browse the directory of patient images.
    - Click button of 'Load'.
4) View-mode.
    - Check meta information in *Status and Patient panel*. 
    - Switch windows in *CT Window Panel* if necessary. 
    - Browse different scans using the *Sliding panel*. 
5) Draw-mode (*Control panel*). 
    - Click 'draw' to activate draw mode. 
    - Left click on images in *Seeds panel* to place red seeds to mark tumors.
    - Right click to switch to green seeds, and then left click on images to mark normal tissues. The changed contour will be shown in *Contour Panel* immediately. 
    - To remove drawed seeds, click 'clear' and drag a rectangle arount the seeds with left click. 
    - Use 'Save' to save the drawed seeds. 
6) Delineate next patient using 'Next' button in 'Load panel'. 
