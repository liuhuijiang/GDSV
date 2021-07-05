# GDSV

The code and dataset for the ACM TOSEM paper Detect Software Vulnerabilities with Weight Biases via Graph Neural Networks, implemented in Tensorflow.

## 1. Environment Setup

The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## 2. Data Preprocessing
The data set of experiment 5 is processed by data_process.The dataset and the ratio of random sampling can be adjusted in the code.

```
python remove_words.py
```

## 3. Usage
To use your own dataset, put the text file under `data/corpus/` and the label file under `data/` as other datasets do. Preprocess the text by running `remove_words.py` before building the graphs.

```
python remove_words.py
```

Build graphs from the datasets in `data/corpus/` as:

```
python build_graph.py
```

Start training as:

```
python train.py
```

## Citation

If you find the code useful, please cite our paper.
```

```

## Contact

For any questions, please drop an email to Shikai Guo.
