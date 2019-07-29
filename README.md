# License Plate Recognition 
This project is for automatic license plate detection and it's number plate recognition. For number plate detection we gathered around 3600 images as training set and manullay annotated all the images. We used yolo version 3 for car detection and license plate detection. For license plate recognition we used CRNN model trained on large set of characters. We tried various other ways to recongize the text such as Google's tesseract , KNN text classification, SVM text classification but the view of plates in provided videos are too far so they were not performing well.


## Installation Requirements For PIP and Anaconda
- Ubuntu 16.04
- Python3.5+
- Pip3  
On fresh Ubuntu use following commands to setup the initial pip3 and other requirements.  
sudo apt-get update  
sudo apt-get install python3-pip  
pip3 install -r requirements.txt  

(Anaconda) conda install numpy opencv matplotlib && conda install pytorch torchvision -c pytorch  

## How to Run on Linux/Mac OS

Download the weights for license plate detection and place it iniside the weights folder.
https://drive.google.com/file/d/1ZJvS_oMr9eAv80HkHMrWgWniYKEPXrXr/view?usp=sharing

To run on video `python3 main_lpr.py --video filename.extension`

## How to Run on RTSP
Open the file main_lpr_RTSP.py and add the urls for all the cameras you want to attach in find_camera function.

Run on RTSP `python3 main_lpr_RTSP.py --camera_id 0`

## Results
We'll add soon.
