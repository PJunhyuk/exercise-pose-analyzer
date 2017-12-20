# exercise-pose-analyzer

## Demo
<img src="/results/shoulder_press_1_pose.gif" width="250">  

[Demo on YouTube - 1](https://youtu.be/ylVMicajB6g)  
[Demo on YouTube - 2](https://youtu.be/VPTsPYFE5Ho)  
[Demo on YouTube - 3](https://youtu.be/vAUZd4LhdFU)  

## Usage

Install [Docker](https://docker.com) and [Kitematic](https://kitematic.com/)

#### Pull docker image
```
$ docker pull jgravity/tensorflow-opencv:odin
$ docker run -it --name odin jgravity/tensorflow-opencv:odin bin/bash
```

> Use ```nvidia-docker``` instead of ```docker``` to use GPU

#### Download/Install code
```
# git clone https://github.com/PJunhyuk/exercise-pose-analyzer
# cd exercise-pose-analyzer
# chmod u+x ./compile.sh && ./compile.sh && cd models/coco && chmod u+x download_models_wget.sh && ./download_models_wget.sh && cd -
```

#### Download sample videos in testset
```
# cd testset && chmod u+x ./download_testset_wget.sh && ./download_testset_wget.sh && cd -
```

#### Just get pose of people
```
# python video_pose.py -f '{video_file_name}'
```
> Qualified supporting video type: mov, mp4

#### Analyze shoulder press
```
# python exercise_analyzer.py -f '{video_file_name}' -e 'sp'
```

###### Arguments
> -f, --videoFile = Path to Video File  
> -w, --videoWidth = Width of Output Video  
> -o, --videoType = Extension of Output Video
> -e, --exerciseType = Type of Exersize  
> - -e 'sp': shoulder press  
> - -e 'dc': dumbbell curl  

## Dependencies

Use Docker [jgravity/tensorflow-opencv](https://hub.docker.com/r/jgravity/tensorflow-opencv/),

or install

- python 3.5.3
- opencv 3.1.0
- jupyter 4.2.1
- git 2.1.4
- tensorflow 1.3.0
- pip packages
  - scipy 0.19.1
  - scikit-image 0.13.1
  - matplotlib 2.0.2
  - pyYAML 3.12
  - easydict 1.7
  - Cython 0.27.1
  - munkres 1.0.12
  - moviepy 0.2.3.2
  - dlib 19.7.0
  - imageio 2.1.2

## Reference

### Citation
    @inproceedings{insafutdinov2017cvpr,
	    title = {ArtTrack: Articulated Multi-person Tracking in the Wild},
	    booktitle = {CVPR'17},
	    url = {http://arxiv.org/abs/1612.01465},
	    author = {Eldar Insafutdinov and Mykhaylo Andriluka and Leonid Pishchulin and Siyu Tang and Evgeny Levinkov and Bjoern Andres and Bernt Schiele}
    }

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
	    booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele}
    }

### Code
[pose-tensorflow](https://github.com/eldar/pose-tensorflow) - Human Pose estimation with TensorFlow framework  
[people-counting-classification](https://github.com/PJunhyuk/people-counting-classification) - Odin: People counting and classification in videos based on pose estimation Edit
