# Team Phoenix
CSCI527 Mahjong AI Agent
- Distribute training reinforcement learning using Ray
- Multi-GPU supervised training
- Built on top of keras and Tensorflow
- Log crawler and viewer included

## Prerequisite
Python 3.7 recommened

```
# install all dependency
pip install -r requirements.txt
```

## Dataset
Logs download from tenhou directly in seqlite format.
You will have to convert it to csv format

### Tenhou log downloader

### Common Issue
Q. No "calculate_shanten" in phoenix_main 

A. istall right version of mahjong. Try ```cd tenhou_env & pip install requirements/lint.txt```

Q. Trainer not found

A. Install phoenix or add root path to PYTHONPATH. ```python setup.py Install```

### Known Issue
* In multi-worker mode, the logs showed some connection issue and the agent exited the game
* Learner runs in low frequency. This is due to the imbalance between Learner's training cycle and cache's fetching samples. The learner is training fast while the cache who is using "ray.get" is running slow.
* Oracle guiding and a good regularizer is needed to improve the RL training
* global reward predictor should be evaluated
