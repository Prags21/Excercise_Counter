# Curls Counter

A real time Biceps curl counter using Streamlit and the Mediapipe's Pose Estimation Model along with OpenCV.

As a beginner, we frequently deal with inappropriate exercise postures, which can result in little benefit or, in the worst-case scenario, undesired muscle injury and strain.
The goal of the counter is to alert the user of insufficient range of motion in real time so that he or she may rectify it without the assistance of a human exercising partner. 


### Working demos

![Demo1](./demo/demo1.png)
![Demo2](./demo/demo2.png)


### Install

A suitable Python 3.x environment with a recent version of Mediapipe is required.
Development and testing was done with Conda Python 3.6.8 on Windows.
All prerequisitites can be found in the `requirements.txt` file. It is adviced that you
install them in a virtual environment.
* Installation in Virtual Environment

   It is recommended that you use [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/)
    to maintain
   a clean Python 3 environment. Create a `virtualenv` and install the requirements:

  ```sh
  $ conda create -n yourenvname python=x.x anaconda

  # To activate it again
  $ source activate yourenvname
            			
  # Install the requirements
  (yourenvname) $ pip3 install -r requirements.txt 
  ```
* System Wide Installation

  ```sh
  pip3 install -r requirements.txt
  ```
  Note: This might change versions of exisiting python packages and hence is *not recommended*.
### Usage

Run the app in conda environment with:
```sh
streamlit run webapp.py
```


### References and Credits

1) [Guide to Human Pose Estimation with Deep Learning(Nanonets)](https://nanonets.com/blog/human-pose-estimation-2d-guide/)
2) [Mediapipe Pose Classification(Google's Github)](https://google.github.io/mediapipe/solutions/pose_classification.html)
3) [Real-time Human Pose Estimation in the Browser(TF Blog)](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
4) [MediaPipePoseEstimation](https://www.youtube.com/watch?v=EgjwKM3KzGU)
