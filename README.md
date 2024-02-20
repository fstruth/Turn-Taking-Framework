# Turn-Taking Framework

In this repo we present a approach to design a general continous model for realizing turn-taking in a human-robot interaction. Due to the lack of training data we propose a framework that don't has to be trained for a specific setting. To make this possible we use rule-based approaches to extend a existing turn-taking model.

We are using the following turn-taking cues and models for our framework:

1. Base: Voice Activity Projection (VAP) by Erik Ekstedt and Gabriel Skantze (https://github.com/ErikEkstedt/VoiceActivityProjection)
2. Headpose
3. Prosodie

This concept is designed for a single user conversation with a robot.

## Information

You can find the paper above in the folder "Paper".

## Technical Requirements

- Number of Cores available: 8
- External Microphone
- External camera

## Use with Furhat

To use the framework with Furhat you need to follow these steps:

1. Design a Skill
2. Implement a ZMQ connection
3. Change the default turn-taking mechanisem of Furhat to use this Framework

## Installation

To use our framework you should install the VAP model first. Therefore we wrote a new installation guide and what to do next.

* Create conda env: `conda create -n turn-taking-framework python=3.11.5`
* PyTorch: `conda install pytorch torchvision torchaudio -c pytorch`
* **VAP Model**
* Install **`Framework`** (this repo):
  * cd to `VoiceActivityProjection` directory and run:
    * `pip install -r requirements.txt`
    * `pip install -e .`
* **Framework**
  * cd to root directory and run:
    * `pip install -r requirements.txt`

## Run

To run the framework make sure you adjusted the processors and have a working ZMQ connection to the Furhat skill.
Make sure that you adjusted the following points in the code:
- `main.py` line 15 - change the audio device 
- `Processing.py` line 75 - adjust the weights and threshold to your personal choice
- `HeadposeDetection.py` line 58 - change the index to your video device
- `HeadposeDetection.py` line 30-50 - calibrate the head movement definitions

Next you simply start the `main.py` and the framework will send a final turn-taking decision to the Furhat skill.

## Outcome & Known Issues

You can find all results and known issues in the paper.
