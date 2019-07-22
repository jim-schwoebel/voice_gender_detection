# voice_gender_detection

As part of the [VOICE Summit](https://www.voicesummit.ai/), I am hosting a machine learning workshop on using the [Voicebook](https://github.com/jim-schwoebel/voicebook). In this workshop, I overview how to train a machine learning model to detect males from females from audio files. 

![](https://media.giphy.com/media/l0HlVq3nJvhSZiZEs/giphy.gif)

## The dataset

I downloaded all the files from [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). After this, I cleaned the data to separate all the males from the females. I took one voice file at random for all the males and females so as to provide unique files.

![](https://github.com/jim-schwoebel/gender-detection/blob/master/data/Screen%20Shot%202019-07-22%20at%2011.16.14%20AM.png)

To prepare the dataset, I put the 'males' and 'females' folders in the data directory of this repository. This will allow for us to featurize the files and train machine learning models via the provided training scripts.

![](https://github.com/jim-schwoebel/gender-detection/blob/master/data/Screen%20Shot%202019-07-22%20at%2012.25.49%20PM.png)

You can download the prepared dataset from [this link](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view).

## Featurization techniques

Intuitively, we know that most of the features that matter for separating out genders are mostly audio-related features like the fundamental frequency, MFCC coeffiicents, and formant frequencies. 

To simplify things, we can just featurize the files with the train_audioclassify.py script, which featurizes audio files with a normalized vector of the first 13 mfcc coefficients and mfcc delta coefficients (in terms of their means, standard deviations, maximum values, and mininum values). Note that I slightly modified this script to include being able to take in .M4A files and converting them to .WAV files. 

```
cd ~
cd desktop/gender-detection
python3 train_audioclassify.py
...
how many classes are you training?2
what is the folder name for class 1?males
what is the folder name for class 2?females
...
ffmpeg version 4.1.3 Copyright (c) 2000-2019 the FFmpeg developers
  built with Apple LLVM version 10.0.1 (clang-1001.0.46.4)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/4.1.3_1 --enable-shared --enable-pthreads --enable-version3 --enable-hardcoded-tables --enable-avresample --cc=clang --host-cflags='-I/Library/Java/JavaVirtualMachines/adoptopenjdk-11.0.2.jdk/Contents/Home/include -I/Library/Java/JavaVirtualMachines/adoptopenjdk-11.0.2.jdk/Contents/Home/include/darwin' --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libbluray --enable-libmp3lame --enable-libopus --enable-librubberband --enable-libsnappy --enable-libtesseract --enable-libtheora --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-librtmp --enable-libspeex --enable-videotoolbox --disable-libjack --disable-indev=jack --enable-libaom --enable-libsoxr
  libavutil      56. 22.100 / 56. 22.100
  libavcodec     58. 35.100 / 58. 35.100
  libavformat    58. 20.100 / 58. 20.100
  libavdevice    58.  5.100 / 58.  5.100
  libavfilter     7. 40.101 /  7. 40.101
  libavresample   4.  0.  0 /  4.  0.  0
  libswscale      5.  3.100 /  5.  3.100
  libswresample   3.  3.100 /  3.  3.100
  libpostproc    55.  3.100 / 55.  3.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '1975.m4a':
  Metadata:
    major_brand     : M4A 
    minor_version   : 512
    compatible_brands: isomiso2
    encoder         : Lavf57.83.100
  Duration: 00:00:14.95, start: 0.000000, bitrate: 79 kb/s
    Stream #0:0(und): Audio: aac (LC) (mp4a / 0x6134706D), 16000 Hz, mono, fltp, 78 kb/s (default)
    Metadata:
      handler_name    : SoundHandler
Stream mapping:
  Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '1975.wav':
  Metadata:
    major_brand     : M4A 
    minor_version   : 512
    compatible_brands: isomiso2
    ISFT            : Lavf58.20.100
    Stream #0:0(und): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s (default)
    Metadata:
      handler_name    : SoundHandler
      encoder         : Lavc58.35.100 pcm_s16le
size=     466kB time=00:00:14.91 bitrate= 256.0kbits/s speed=1.45e+03x    
video:0kB audio:466kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.016346%
MALES - featurizing 1975.wav
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/scipy/linalg/basic.py:1226: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
  warnings.warn(mesg, RuntimeWarning)
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/scipy/signal/_arraytools.py:45: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  b = a[a_slice]
making 0.wav
[-2.61770391e+02  6.38850847e+01 -4.61535563e+02 -1.06573575e+02
  1.36820101e+02  4.72779880e+01  6.68131896e+00  2.21282472e+02
 -1.51429929e+01  2.62338730e+01 -9.80571152e+01  5.61283764e+01
  5.65516843e+01  3.01895657e+01 -9.57117883e+00  1.46959211e+02
 -1.91317383e+01  2.11478167e+01 -7.56840487e+01  3.09956732e+01
  3.18617544e+01  2.10894590e+01 -3.01884536e+01  9.63904368e+01
 -2.13284629e+01  1.88760859e+01 -7.17951960e+01  1.85254957e+01
  1.49474937e+01  1.42591915e+01 -2.50579532e+01  5.78429200e+01
 -8.90644306e+00  1.11660257e+01 -5.46698635e+01  1.85997678e+01
 -1.97143447e+00  1.04581202e+01 -3.72414734e+01  2.44524712e+01
  1.67548769e+00  1.14801833e+01 -3.15460685e+01  3.20846052e+01
 -2.53759183e+00  9.96187581e+00 -3.94097288e+01  2.37449071e+01
  5.64976595e+00  7.78199113e+00 -2.06583351e+01  2.97178817e+01
 -2.22377091e-01  1.26072071e+01 -3.09844196e+01  4.07686153e+01
  1.41587253e-02  9.27049668e+00 -2.69449668e+01  2.44447366e+01
  3.91153066e-02  5.21001556e+00 -1.46309269e+01  1.45735832e+01
 -4.62229897e-02  6.18090719e+00 -1.47701417e+01  2.22470109e+01
  1.93623371e-02  4.19694354e+00 -1.27877006e+01  9.39109635e+00
  3.56801243e-03  3.96213573e+00 -1.37006813e+01  8.99878932e+00
  6.26321844e-02  3.39339019e+00 -1.06445450e+01  9.18719073e+00
 -9.01324696e-03  2.68399923e+00 -9.23715568e+00  6.21079082e+00
  2.60406242e-02  2.04737481e+00 -6.55178962e+00  6.33710873e+00
 -2.67108609e-02  2.02573621e+00 -6.16472991e+00  5.69279759e+00
 -1.96293894e-02  2.17288029e+00 -6.86314894e+00  5.51207613e+00
  4.94949583e-03  1.85195145e+00 -5.32477863e+00  5.55641781e+00
 -1.48860490e-02  1.55405927e+00 -4.85733502e+00  3.94697571e+00
 -9.63985246e+00  2.95645368e+00 -1.54140254e+01 -4.86997792e+00
  3.77656232e+00  1.50605849e+00  1.18451486e+00  6.07380916e+00
 -3.84191553e-01  8.92672845e-01 -2.22380388e+00  1.26244831e+00
  2.34779052e+00  9.99468618e-01  7.27284681e-01  4.24105022e+00
 -7.70760906e-01  5.76365815e-01 -1.82246750e+00  1.72374958e-01
  1.62145416e+00  6.36923142e-01  6.86800470e-01  3.23502968e+00
 -7.03139457e-01  7.70496449e-01 -2.16410156e+00  4.04966447e-01
  7.02896057e-01  3.52589832e-01 -1.55512950e-01  1.35459749e+00
 -1.54157297e-01  3.76350657e-01 -6.76922065e-01  5.52239968e-01
  1.31166314e-01  3.95427560e-01 -7.08800989e-01  6.91590489e-01
  2.69257759e-01  2.92018650e-01 -4.68028712e-01  8.03338235e-01
 -9.43629015e-02  2.69322187e-01 -5.89132534e-01  4.46403512e-01
  2.64683353e-01  2.74896394e-01 -1.95170047e-01  8.22367853e-01
  1.56588762e-01  8.18398590e-01 -8.69151341e-01  1.23325537e+00
  4.32778914e-02  4.26126780e-01 -3.80986657e-01  6.59090643e-01
  1.12143640e-01  1.48501333e-01 -1.54953287e-01  3.31785097e-01
 -1.10071038e-01  2.21420052e-01 -4.13527492e-01  2.43018483e-01
 -2.57906280e-02  1.24994764e-01 -2.71912676e-01  1.82642733e-01
  6.79165063e-02  8.71738334e-02 -5.81913857e-02  2.79734470e-01
 -5.32248472e-03  2.00736996e-01 -3.56823051e-01  2.39871577e-01
 -2.72560162e-02  3.07670925e-02 -6.64212534e-02  4.72707180e-02
  2.90053187e-02  9.17872166e-02 -1.33647569e-01  1.67763803e-01
 -3.52328817e-02  9.34423499e-02 -1.58491671e-01  1.43928428e-01
 -1.79746122e-02  5.97522043e-02 -1.02999337e-01  7.28779588e-02
  1.83742387e-02  6.78073641e-02 -5.76510823e-02  1.13251858e-01
 -4.75863543e-02  2.51204094e-02 -8.20748155e-02  5.02732595e-03]
...
```

After running the script, all the audio files are appropriately featurized into .JSON files (e.g. 1.wav --> 1.json) into feature arrays like this:

```
{"features": [-353.90257942328554, 58.01672170639758, -703.2908063668561, -244.59664968903266, 120.77438799548838, 48.587638379552686, 0.0, 206.23937949197023, -17.614903082924762, 21.896880584719494, -93.53524512304182, 39.25891412027272, 71.5585736549912, 31.216641866853976, -4.1097497409468655, 147.80726459654613, -11.212912690856928, 26.241167154558216, -93.3273860041447, 33.39406296150048, 17.385199406906743, 17.812256829651783, -15.775957864593206, 71.2976899178332, -27.475327059655772, 17.29486585997999, -72.30450274808385, 16.555004086550632, 1.3154971178660386, 15.045541210536745, -34.91050791010009, 36.23435893500907, -10.736291647533783, 11.850996840148898, -38.55204075924968, 17.08622970202298, -7.794566613463251, 9.080422143454681, -32.85545573295984, 14.14811618776311, -10.1821346272071, 9.596484081862144, -42.976800137609224, 18.983509328321023, -10.085169587133, 7.838356955314714, -29.87348386201144, 15.39133575524436, 2.21927290207588, 6.888356028816847, -19.172128831473927, 18.15741693999623, -0.9291675352603528, 12.576850016655293, -37.051447993039474, 32.92362919622146, 0.10573689916014054, 10.789639202489477, -23.490694209053974, 24.671154779939425, -0.01629452157074945, 4.433766495443947, -11.142467331094984, 15.493827623920136, 0.17573108461133594, 6.427849955456987, -16.826745664570108, 19.093468482334636, 0.4173772558279777, 5.551845120120025, -13.96684516782555, 14.647753711947907, -0.13604156159188882, 3.838525192356469, -10.242221290651118, 9.865500478056386, 0.18312786648656101, 3.713931812887859, -8.746542201847214, 7.039678136997143, -0.1156850991593573, 3.331736740379702, -8.073791330947259, 8.742059331786745, 0.1943999961523158, 1.928274091002802, -6.289087093770005, 4.253015942505109, -0.15227882868669954, 1.7794453105311643, -4.985202596317516, 4.458126480445773, -0.01990872747521754, 1.7621594175996373, -5.507148800360761, 5.600782235826579, 0.031014894536071806, 1.1640592421749514, -2.728100820895949, 3.518347326841714, -0.05704584862225418, 1.3921398794420592, -3.9409657702177654, 3.489564362897586, -40.151813144983024, 5.950517472992894, -49.34503450029317, -28.683742793608165, 15.727627542142779, 7.797730402032888, -1.0491314918340835, 25.77992243649628, -1.8481648389201435, 2.797469322091307, -6.725810547443524, 3.2079518282940693, 6.806005243735187, 3.7964095386621173, 0.8864980798360538, 14.78516612300923, -2.7535395662737647, 3.4489392434551163, -7.405200391272226, 3.635880010797687, 2.1522411143011846, 1.8879916910378791, -0.07112269107952354, 6.30224475627656, -4.675803858738777, 2.532142530211178, -9.038062843510481, 0.0420118619989896, -0.30375097359594255, 2.9110433901249833, -4.1552391436218965, 4.529294866876134, -2.0540693105317085, 1.3461096219830693, -3.9712222862092643, 0.6630461376529606, -0.9157642418498797, 1.0097019614734404, -2.3922264545952787, 0.7673701453082771, -1.7796014228347945, 1.4539519485762857, -5.372100017201153, 0.43555065900440093, -1.8461844818049344, 1.041907648535014, -3.8606951077969747, 0.34998386129101644, -0.107616150299151, 1.1046699346585973, -2.396516103934241, 1.5096538177891154, 0.6347088123470858, 1.0892073311101835, -1.2553296114221593, 2.5230047469160866, 0.1722624017918903, 1.8518190235596568, -2.398730145651318, 3.083894347492428, 0.01301137686073959, 0.5349383203841249, -0.9954767461611344, 0.7837855794519846, 0.46160195048517966, 0.7040488660965134, -1.0060353980563808, 1.2298063908674715, 0.2543028665321045, 0.5518685910277583, -0.34013348915424313, 1.4529802275102828, -0.13012888884825968, 0.31820989899446706, -0.7697971157550454, 0.24009171929379497, 0.14263686001849904, 0.5712258640566474, -1.015970774725115, 0.7810490874346672, -0.2711011653441329, 0.3916818703065981, -1.0092239163684074, 0.21513224077628454, 0.14348795775826292, 0.23612641254819336, -0.13031067424234943, 0.5316269928131386, -0.0992375489967987, 0.21747840902867793, -0.43557271818530324, 0.1418337399826225, -0.1344329753837571, 0.22892381923790125, -0.6883936000450951, 0.13727461110098693, -0.06237816425870175, 0.223239850355424, -0.33660386623955496, 0.43979341585521425, -0.023783173746338017, 0.2778740777883613, -0.4926207212772207, 0.43619554536219823]}
```
These .JSON files have the labels 

```
labels = ['mfcc1_mean_(0.02 second window)', 'mfcc1_std_(0.02 second window)', 'mfcc1_max_(0.02 second window)', 'mfcc1_min_(0.02 second window)', 'mfcc2_mean_(0.02 second window)', 'mfcc2_std_(0.02 second window)', 'mfcc2_max_(0.02 second window)', 'mfcc2_min_(0.02 second window)', 'mfcc3_mean_(0.02 second window)', 'mfcc3_std_(0.02 second window)', 'mfcc3_max_(0.02 second window)', 'mfcc3_min_(0.02 second window)', 'mfcc4_mean_(0.02 second window)', 'mfcc4_std_(0.02 second window)', 'mfcc4_max_(0.02 second window)', 'mfcc4_min_(0.02 second window)', 'mfcc5_mean_(0.02 second window)', 'mfcc5_std_(0.02 second window)', 'mfcc5_max_(0.02 second window)', 'mfcc5_min_(0.02 second window)', 'mfcc6_mean_(0.02 second window)', 'mfcc6_std_(0.02 second window)', 'mfcc6_max_(0.02 second window)', 'mfcc6_min_(0.02 second window)','mfcc7_mean_(0.02 second window)', 'mfcc7_std_(0.02 second window)', 'mfcc7_max_(0.02 second window)', 'mfcc7_min_(0.02 second window)', 'mfcc8_mean_(0.02 second window)', 'mfcc8_std_(0.02 second window)', 'mfcc8_max_(0.02 second window)', 'mfcc8_min_(0.02 second window)', 'mfcc9_mean_(0.02 second window)', 'mfcc9_std_(0.02 second window)', 'mfcc9_max_(0.02 second window)', 'mfcc9_min_(0.02 second window)', 'mfcc10_mean_(0.02 second window)', 'mfcc10_std_(0.02 second window)', 'mfcc10_max_(0.02 second window)', 'mfcc10_min_(0.02 second window)', 'mfcc11_mean_(0.02 second window)', 'mfcc11_std_(0.02 second window)', 'mfcc11_max_(0.02 second window)', 'mfcc11_min_(0.02 second window)', 'mfcc12_mean_(0.02 second window)', 'mfcc12_std_(0.02 second window)', 'mfcc12_max_(0.02 second window)', 'mfcc12_min_(0.02 second window)', 'mfcc13_mean_(0.02 second window)', 'mfcc13_std_(0.02 second window)', 'mfcc13_max_(0.02 second window)', 'mfcc13_min_(0.02 second window)', 'mfccdelta1_mean_(0.02 second window)', 'mfccdelta1_std_(0.02 second window)', 'mfccdelta1_max_(0.02 second window)', 'mfccdelta1_min_(0.02 second window)', 'mfccdelta2_mean_(0.02 second window)', 'mfccdelta2_std_(0.02 second window)', 'mfccdelta2_max_(0.02 second window)', 'mfccdelta2_min_(0.02 second window)', 'mfccdelta3_mean_(0.02 second window)', 'mfccdelta3_std_(0.02 second window)', 'mfccdelta3_max_(0.02 second window)', 'mfccdelta3_min_(0.02 second window)','mfccdelta4_mean_(0.02 second window)', 'mfccdelta4_std_(0.02 second window)', 'mfccdelta4_max_(0.02 second window)','mfccdelta4_min_(0.02 second window)', 'mfccdelta5_mean_(0.02 second window)', 'mfccdelta5_std_(0.02 second window)','mfccdelta5_max_(0.02 second window)', 'mfccdelta5_min_(0.02 second window)', 'mfccdelta6_mean_(0.02 second window)', 'mfccdelta6_std_(0.02 second window)', 'mfccdelta6_max_(0.02 second window)', 'mfccdelta6_min_(0.02 second window)', 'mfccdelta7_mean_(0.02 second window)', 'mfccdelta7_std_(0.02 second window)', 'mfccdelta7_max_(0.02 second window)', 'mfccdelta7_min_(0.02 second window)', 'mfccdelta8_mean_(0.02 second window)', 'mfccdelta8_std_(0.02 second window)', 'mfccdelta8_max_(0.02 second window)', 'mfccdelta8_min_(0.02 second window)', 'mfccdelta9_mean_(0.02 second window)', 'mfccdelta9_std_(0.02 second window)', 'mfccdelta9_max_(0.02 second window)', 'mfccdelta9_min_(0.02 second window)', 'mfccdelta10_mean_(0.02 second window)', 'mfccdelta10_std_(0.02 second window)', 'mfccdelta10_max_(0.02 second window)', 'mfccdelta10_min_(0.02 second window)', 'mfccdelta11_mean_(0.02 second window)', 'mfccdelta11_std_(0.02 second window)', 'mfccdelta11_max_(0.02 second window)', 'mfccdelta11_min_(0.02 second window)', 'mfccdelta12_mean_(0.02 second window)', 'mfccdelta12_std_(0.02 second window)', 'mfccdelta12_max_(0.02 second window)', 'mfccdelta12_min_(0.02 second window)', 'mfccdelta13_mean_(0.02 second window)', 'mfccdelta13_std_(0.02 second window)', 'mfccdelta13_max_(0.02 second window)', 'mfccdelta13_min_(0.02 second window)', 'mfcc1_mean_(0.50 second window)', 'mfcc1_std_(0.50 second window)', 'mfcc1_max_(0.50 second window)', 'mfcc1_min_(0.50 second window)', 'mfcc2_mean_(0.50 second window)', 'mfcc2_std_(0.50 second window)', 'mfcc2_max_(0.50 second window)', 'mfcc2_min_(0.50 second window)', 'mfcc3_mean_(0.50 second window)', 'mfcc3_std_(0.50 second window)', 'mfcc3_max_(0.50 second window)', 'mfcc3_min_(0.50 second window)', 'mfcc4_mean_(0.50 second window)', 'mfcc4_std_(0.50 second window)', 'mfcc4_max_(0.50 second window)', 'mfcc4_min_(0.50 second window)', 'mfcc5_mean_(0.50 second window)', 'mfcc5_std_(0.50 second window)', 'mfcc5_max_(0.50 second window)', 'mfcc5_min_(0.50 second window)', 'mfcc6_mean_(0.50 second window)', 'mfcc6_std_(0.50 second window)', 'mfcc6_max_(0.50 second window)', 'mfcc6_min_(0.50 second window)', 'mfcc7_mean_(0.50 second window)', 'mfcc7_std_(0.50 second window)', 'mfcc7_max_(0.50 second window)', 'mfcc7_min_(0.50 second window)','mfcc8_mean_(0.50 second window)', 'mfcc8_std_(0.50 second window)', 'mfcc8_max_(0.50 second window)', 'mfcc8_min_(0.50 second window)','mfcc9_mean_(0.50 second window)', 'mfcc9_std_(0.50 second window)', 'mfcc9_max_(0.50 second window)', 'mfcc9_min_(0.50 second window)', 'mfcc10_mean_(0.50 second window)', 'mfcc10_std_(0.50 second window)', 'mfcc10_max_(0.50 second window)', 'mfcc10_min_(0.50 second window)', 'mfcc11_mean_(0.50 second window)', 'mfcc11_std_(0.50 second window)', 'mfcc11_max_(0.50 second window)', 'mfcc11_min_(0.50 second window)', 'mfcc12_mean_(0.50 second window)', 'mfcc12_std_(0.50 second window)', 'mfcc12_max_(0.50 second window)', 'mfcc12_min_(0.50 second window)', 'mfcc13_mean_(0.50 second window)', 'mfcc13_std_(0.50 second window)', 'mfcc13_max_(0.50 second window)', 'mfcc13_min_(0.50 second window)', 'mfccdelta1_mean_(0.50 second window)', 'mfccdelta1_std_(0.50 second window)', 'mfccdelta1_max_(0.50 second window)', 'mfccdelta1_min_(0.50 second window)', 'mfccdelta2_mean_(0.50 second window)', 'mfccdelta2_std_(0.50 second window)', 'mfccdelta2_max_(0.50 second window)', 'mfccdelta2_min_(0.50 second window)', 'mfccdelta3_mean_(0.50 second window)', 'mfccdelta3_std_(0.50 second window)', 'mfccdelta3_max_(0.50 second window)', 'mfccdelta3_min_(0.50 second window)', 'mfccdelta4_mean_(0.50 second window)', 'mfccdelta4_std_(0.50 second window)', 'mfccdelta4_max_(0.50 second window)', 'mfccdelta4_min_(0.50 second window)', 'mfccdelta5_mean_(0.50 second window)', 'mfccdelta5_std_(0.50 second window)', 'mfccdelta5_max_(0.50 second window)', 'mfccdelta5_min_(0.50 second window)', 'mfccdelta6_mean_(0.50 second window)', 'mfccdelta6_std_(0.50 second window)', 'mfccdelta6_max_(0.50 second window)', 'mfccdelta6_min_(0.50 second window)', 'mfccdelta7_mean_(0.50 second window)', 'mfccdelta7_std_(0.50 second window)', 'mfccdelta7_max_(0.50 second window)', 'mfccdelta7_min_(0.50 second window)', 'mfccdelta8_mean_(0.50 second window)', 'mfccdelta8_std_(0.50 second window)', 'mfccdelta8_max_(0.50 second window)', 'mfccdelta8_min_(0.50 second window)', 'mfccdelta9_mean_(0.50 second window)', 'mfccdelta9_std_(0.50 second window)', 'mfccdelta9_max_(0.50 second window)', 'mfccdelta9_min_(0.50 second window)', 'mfccdelta10_mean_(0.50 second window)', 'mfccdelta10_std_(0.50 second window)', 'mfccdelta10_max_(0.50 second window)', 'mfccdelta10_min_(0.50 second window)', 'mfccdelta11_mean_(0.50 second window)', 'mfccdelta11_std_(0.50 second window)', 'mfccdelta11_max_(0.50 second window)', 'mfccdelta11_min_(0.50 second window)', 'mfccdelta12_mean_(0.50 second window)', 'mfccdelta12_std_(0.50 second window)', 'mfccdelta12_max_(0.50 second window)', 'mfccdelta12_min_(0.50 second window)', 
```
## Modeling techniques 

### using train_audioclassify.py
Originally, after training with train_audioclassify.py. Note that the classes were auto-balanced randomly to build a machine learning model between the groups (2312 males / 2312 females). 

```
MALES is 1371 more than min value, balancing...
MALES is 1370 more than min value, balancing...
MALES is 1369 more than min value, balancing...
MALES is 1368 more than min value, balancing...
MALES is 1367 more than min value, balancing...
MALES is 1366 more than min value, balancing...
MALES is 1365 more than min value, balancing...
MALES is 1364 more than min value, balancing...
MALES is 1363 more than min value, balancing...
```
The output accuracy achieved is as follows:

```
Decision tree accuracy (+/-) 0.007327676542764603
0.7398596519424567
Gaussian NB accuracy (+/-) 0.016660391044338484
0.8682797740896762
SKlearn classifier accuracy (+/-) 0.00079538963465451
0.5157270607408913
Adaboost classifier accuracy (+/-) 0.013940745120583124
0.8892763651333413
Gradient boosting accuracy (+/-) 0.01950292233912751
0.8669747415791165
Logistic regression accuracy (+/-) 0.012678238150779661
0.894515837971657
Hard voting accuracy (+/-) 0.013226860908589952
0.9076178049591996
K Nearest Neighbors accuracy (+/-) 0.017244722910655787
0.731352177051436
Random forest accuracy (+/-) 0.02258623279374182
0.8079923672086033
svm accuracy (+/-) 0.022841304608332974
0.8781480823563248
most accurate classifier is Hard Voting with audio features (mfcc coefficients).
saving classifier to disk
summarizing session...
VotingClassifier(estimators=[('gradboost', GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=1.0, loss='deviance', max_depth=1,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_l...='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=None))],
         flatten_transform=None, n_jobs=1, voting='hard', weights=None)
['hard voting', 0.9076178049591996, 0.013226860908589952]
```

### using train_audioTPOT.py 

I also tried training a model with the train_audioTPOT.py script. Note you need to run this script after already running the train_audioclassify.py script for it to work properly, as the train_audioclassify.py script featurizes the audio files appropriately. 

```
Jims-MacBook-Pro:gender-detection jimschwoebel$ python3 train_audioTPOT.py
classification (c) or regression (r) problem? 
c
what is the name of class 1? 
males
what is the name of class 2? 
females
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Optimization Progress:  11%|█▏         | 34/300 [08:07<1:11:05, 16.03s/pipelin
```

A summary of the accuracy of the model is below. Note that the train_audioTPOT script sometimes fails when executing, so you should keep this in mind and perhaps switch computers if it isn't training properly.

### using train_audiokeras.py 

I also tried training with a bare-bones keras MLP. This doesn't do anything special, it's just a very simple neural network to see how deep learning performs on the dataset. In the future, it would make sense to fine-tune this network based on peer-reviewed publications (some have achieved up to 98% accuracy). 

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

## Learn more
Any feedback on this repository is greatly appreciated. 
* Learn more about voice computing with the textbook I wrote [Introduction to Voice Computing in Python](https://github.com/jim-schwoebel/voicebook).
* If you'd like to be mentored by someone on our team, check out the [Innovation Fellows Program](http://neurolex.ai/research).
* If you want to talk to me directly, please send me an email @ js@neurolex.co. 

## References
* [prepared dataset](https://drive.google.com/file/d/1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO/view)
* [VOICE Summit](https://www.voicesummit.ai/)
* [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
