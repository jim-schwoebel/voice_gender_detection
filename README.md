# gender-detection

As part of the VOICE Summit, I am hosting a machine learning workshop on using the [Voicebook](https://github.com/jim-schwoebel/voicebook) to train a machine learning model to detect males from females. 

![](https://media.giphy.com/media/3oEjHPqljqeuYPLOTK/giphy.gif)

## The dataset

I downloaded all the files from [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). After this, I cleaned the data to separate all the males from the females. I took one voice file at random for all the males and females so as to provide unique files.

![]()

You can download the prepared dataset from [this link](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view).

## Featurization techniques

Intuitively, we know that most of the features that matter for separating out genders are mostly audio-related features like the fundamental frequency, MFCC coeffiicents, and formant frequencies. 

As a first pass, we can use some of the feature arrays of the [Voicebook](). 

## Modeling techniques 

I used the train_audioTPOT.py script to train the model.

## References
* [prepared dataset](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view)
* [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
