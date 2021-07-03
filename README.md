# GDSV

Code for our ACM TOSEM paper,

## 1. Environment Setup

The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## 2. Data Preprocessing

## 3. Vocab & Data Preparation

```
cd generator/tranlator
sh prepare.sh # check it before use
```

## 4. Train

```
python remove_words.py
python build_graph.py
python train.py
```
