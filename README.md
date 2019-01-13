# face-recognition
Implementation and comparision of face recognition algorithms.

This project was developed for the Numerical Linear Algebra discipline at UERJ/IPRJ under orientaion of Professor Ricardo Fabbri.

## Authors
Julia de Araújo Nascimento, Thiago Franco, Vítor Gavião

## Running
Install the project requirements by running 

`pip install -r requirements.txt`

To evaluate a model, run

`python face_recognition.py eval [path to dataset directory]`

e.g. `python face_recognition.py eval ./datasets/easy`

To recognize an image based on a model for a given dataset, run

`python face_recognition.py recl [path to dataset directory] [path to image to be recognized]`

For a more interactive experience, launch the `jupyter notebook` and enjoy :) 

Alternatively, try lauching the notebook through Binder (it may take a while to load):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thiago-franco/face-recognition/master)
