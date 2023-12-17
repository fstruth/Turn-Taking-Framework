# Turn-Taking Framework

We are using the following turn-taking cues and models for our framework:

1. Prosodie
2. VAP
3. Head Pose

## Information

Link to Paper

## Use

How to use it with Furhat

## Installation

To use our framework you should install the VAP model first. Therefore we wrote a new installation guide and what to do after that.

* Create conda env: `conda create -n voice_activity_projection python=3.11.5`
* PyTorch: `conda install pytorch torchvision torchaudio -c pytorch`
* **VAP Model**
* Install **`Framework`** (this repo):
  * cd to `VoiceActivityProjection` directory and run:
    * `pip install -r requirements.txt`
    * `pip install -e .`
* **Framework**
  * cd to root directory and run:
    * `pip install -r requirements.txt`

