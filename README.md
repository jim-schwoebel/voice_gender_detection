# gender-detection

As part of the [VOICE Summit](https://www.voicesummit.ai/), I am hosting a machine learning workshop on using the [Voicebook](https://github.com/jim-schwoebel/voicebook). In this workshop, I overview how to train a machine learning model to detect males from females. 

![](https://media.giphy.com/media/l0HlVq3nJvhSZiZEs/giphy.gif)

## The dataset

I downloaded all the files from [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). After this, I cleaned the data to separate all the males from the females. I took one voice file at random for all the males and females so as to provide unique files.

![](https://github.com/jim-schwoebel/gender-detection/blob/master/data/Screen%20Shot%202019-07-22%20at%2011.16.14%20AM.png)

You can download the prepared dataset from [this link](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view).

## Featurization techniques

Intuitively, we know that most of the features that matter for separating out genders are mostly audio-related features like the fundamental frequency, MFCC coeffiicents, and formant frequencies. 

To simplify things, we can just featurize the files with the train_audioclassify.py script. I slighly modified this script to include being able to take in .M4A files and converting them to .WAV files.

## Modeling techniques 

I used the train_audioTPOT.py script to train the model.

## Making model predictions 

All you need to do to make a model prediction is to provide an audio file from the command line. Note that the audio file must be a .WAV file in order for it to make a proper prediction.

```
predict.py test.wav
```

This will look for the file test.wav in the current directory, featurize the file, and then make a model prediction appropriately. If you'd like to save this model prediction as .JSON, feel free to pass through another argument at the end.

```
predict.py test.wav yes
```

This will featurize the file test.wav and save the model prediction in 'test.json.' 

## References
* [prepared dataset](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view)
* [VOICE Summit](https://www.voicesummit.ai/)
* [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
