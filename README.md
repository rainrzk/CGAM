<h1 align="center">Classifier-GAN Arcaea Mapper</h1>

<p align="center">
    <em>~a simple auto-mapper for arcaea~</em>
</p>

This work assumes rhythm and geometry are independently separable. Part of the idea comes from [Flow-Gan](https://github.com/kotritrona/osumapper). The architecture is implemented in [TensorFlow](https://github.com/tensorflow/tensorflow). Python 3.12.2 is recommended. If you are new to mapping or don't know how to preview a chart, check [ArcCreate](https://github.com/Arcthesia/ArcCreate).

## How to Use

### Train

Prepare `dataset/` like follows:

```shell
dataset/
    song1/
        2.aff
        3.aff
        base.ogg
    song2/
        3.aff
        base.ogg
    ...
```

Install `requirements.txt` before running `train.ipynb`:

```shell
pip install -r requirements.txt
```

Thanks for the charter:3

### Inference

Prepare `models/` like follows:

```shell
models/
    gan_discriminator.keras
    gan_generator.keras
    note_classifier_model.keras
```

Setup the path of your audio and skeleton aff (*i.e.* an aff that only consists of `AudioOffset` and `timing`), for example:

```shell
inputs/
    3.aff
    base.ogg
```

Then run `inference.ipynb`.

Other configurations are in `config.yaml`, feel free to adjust epochs or other hyperparameters that fit your dataset.

## Why It Works

### Pipeline

The pipeline grows as follows.

<div align="center">
    <img src="./pipeline.svg" width=90%>
</div>

### Note Classifier

The Note Classifier is only used for rhythm (*i.e.* the start/end timestamp of any notes in a chart).

<div align="center">
    <img src="./noteclassifier.svg" width=90%>
</div>

### GAN

The GAN is only used for geometry (*i.e.* the lane/coordinate of any notes in a chart).

<div align="center">
    <img src="./gan.svg" width=90%>
</div>

## FAQ

- Q1: How do you implement Grid Divisor?
    - This idea comes from [polytone](https://impactstd.co/). Instead of the time-length of a note, giving each legal note (*i.e.* 4/8/12/16/24/32nd tick) a grid position is better. There is a penalty module penalizes weird rhythm patterns (*i.e.* 24th mix 32nd).

- Q2: Is each timestamp independent?
    - No. Because of the Grid Divisor, every measure is set to be a unit sample. That is, inside a measure - the order of ticks is distinguishable. Each measure is almost independent, with a context window only from the previous measure.

- Q3: Does this classifier work once for every legal timestamp from the grid line?
    - Yes. For every timestamp, the classifier predicts if there exists a note starts, how many notes start and if it's inside the duration of a longnote.

- Q4: Is this architecture unable to predict more than 3 longnotes or arctaps at the same timestamp?
    - Yes. But this is adjustable - feel free to change the dimension of paramlist as you like.

- Q5: How do you avoid pattern-shrinkage in GANs?
    - So there is a 5% data augmentation process before training.

- Q6: Is this a conditional or aggregational GAN?
    - Not yet, it is a simple GAN, but it could be conditional in the future. The mask sent from Note Classifier only decides the timestamp used to train GAN. GAN still generates the full-length geometry of ticks in a measure, who doesn't and shouldn't know when to have a note.

- Q7: What is the algorithm of max finger count?
    - Firstly perform a threshold detection to discard some types of notes, secondly calculate a score for candidate note-types, thirdly use a top-K algorithm (including inside the duration of longnotes) to ensure not greater than max finger count.

- Q8: How do you ensure the output range is expected?
    - For lanes, round to 1,2,3,4. For coordinates, normalize to (0.5±1,0.5±0.5) then crop to trapezoid and assign them to nearest (Δ0.25,Δ0.25) coordinate grids. This is modifiable.

## What to Do Next

*Make the thresholds of top-K algorithm learnable...so won't need to adjust them frequently to ensure not always generating taps or always generating arcs*

*Think of a better way rather than normalizing coordinates...GANs seem to mimic the mean very badly*

*I wanted to add a memoization module connected to Note Classifier and GAN, in simple words it could be a momentum...aw I'm too lazy...*
