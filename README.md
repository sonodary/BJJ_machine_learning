# Brazilian Jiu-Jitsu Image Classification Project
This repository contains all of our work for our project on BJJ position classification using Neural Networks. 

## Goal
[Brazilian Jiu-Jitsu](https://www.youtube.com/watch?v=3Ef_uYF7ABw&ab_channel=FloGrappling) is the world's foremost submission grappling sport. For our project, we aim to create a machine learning model that is able to accurately classify BJJ positions from images of two competitors. We will be using [this](https://vicos.si/resources/jiujitsu/) dataset of images of two athletes doing BJJ.

## Dataset
This dataset contains 120,279 labeled images of 2 jiu-jitsu athletes sparring in different combat positions. The images all have one of the following 18 labels:
- `standing`
- `takedown1` or `takedown2`
- `open_guard1` or `openguard2`
- `half_guard1` or `half_guard2`
- `closed_guard1` or `closed_guard2`
- `5050_guard`
- `mount1` or `mount2`
- `back1` or `back2`
- `turtle1` or `turtle2`
- `side_control1` or `side_control2`

Where `1` or `2` indicates which of the athletes is in the specified position and the absence of a number indicates that both athletes are in the position. Due to the nature of BJJ, some positions require the numbers and others don't. These labels cover the most common positions in BJJ, however some of them are a little broad in how many positions they cover. Thus, a good addition to this dataset would be a more comprehensive labeling system.

## Models
We strive to create two types of Neural Networks for this problem: a Convolutional Neural Network and a Recurrent Neural Network. We will create these two types of NNs as we believe that these are the most suited for the task at hand. One of our goals is also to compare the performance of both of these models at classifying the images. If we have time, we would also like to compare the performance of our models with a "_simpler_" model, like a Support Vector Machine.


