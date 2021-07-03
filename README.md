# GDSV

Code for our ACM TOSEM paper,

## 1. Environment Setup

The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## 2. Data Preprocessing
The data set of experiment 5 is processed by data_process.

To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. Preprocess the text by running `remove_words.py` before building the graphs.

```
python data_process.py
```

## 4. Train

```
python remove_words.py
python build_graph.py
python train.py
```

## Citation

If you find the code useful, please cite our paper.
```

```

## Contact

For any questions, please drop an email to [Shikai Guo](https://jcyk.github.io/).
