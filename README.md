This is my adventure to play with tensorflow stuff.
I am using ZED camera to take pictures of dice and hopefully train GoogleNet to predict rolled value. Not using ZED stereo vision so any webcam will do. Just using it's simple api.

Run stuff like this: 
python3 app.py --model_name=noppa --label_size=6
python3 train.py --model_name=noppa --label_size=6
