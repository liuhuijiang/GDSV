import json_lines
import json
import re

def count_words(aaa):
    # 计算单词数
    # word=[]
    # str_list=aaa.split()# 代码单词分开，分隔符默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等，返回一个列表
    # for i in range(len(str_list)):
    #     if str_list[i] in word:
    #         continue
    #     else:
    #         word.append(str_list[i])
    # return len(word)
    str_list = aaa.split()  # 代码单词分开，分隔符默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等，返回一个列表
    return len(str_list)
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string.replace("\n", "")
    string.replace("->"," -> ")

    string = re.sub(r"[^A-Za-z0-9()%/\\:+\->&\[\]|=<*>.,_{};!?\'\`]", " ", string)      # 实现正则的替换，^匹配开始位置，匹配数字，字母，下划线，（）
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"<<", " << ", string)
    string = re.sub(r">>", " >> ", string)
    string = re.sub(r"&", " & ", string)
    string = re.sub(r"/\*", " /* ", string)
    # string = re.sub(r"->", " -> ", string)
    string = re.sub(r"../../data_process", " . ", string)
    string = re.sub(r"\*/", " */ ", string)
    string = re.sub(r"\*", " * ", string)
    string = re.sub(r"&&", " && ", string)  # 新添加的&&
    string = re.sub(r":", " : ", string)
    # string = re.sub(r"\||", " || ", string)    # 新添加的||
    string = re.sub(r";", " ; ", string)    # 新添加的；

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\{", " { ", string)    # 新添加的{
    string = re.sub(r"}", " } ", string)    # 新添加的}
    string = re.sub(r"'", " ' ", string)    # 新添加的'
    string = re.sub(r"\+", " + ", string)    # 新添加的+
    string = re.sub(r"-", " - ", string)       # 新添加的->
    string = re.sub(r">", " > ", string)
    string = re.sub(r"<", " < ", string)
    string = re.sub(r"\s{2,}", " ", string)     # 匹配任意空白字符，2到无穷次，只用一个‘ ’替换掉

    return string.strip()

# def clean_str(string):
#     return string.strip()

line_num = -1
num0 = 0
num1 = 0
dataset = "FFmpeg+qemu"   # FFmpeg , qemu ，VDSIC
fw = open(dataset+'_dataset.txt', 'w+', encoding='utf-8')
fl = open(dataset+'_datalabel.txt', 'w+', encoding='utf-8')
t = 0
with open(dataset+'_train.jsonl', 'rb') as frain:
    for item in json_lines.reader(frain):
        func_str = clean_str(item.get("func"))  # 对函数代码进行预处理
        # func_str = item.get("func")
        t = t + 1
        if item.get("target") == 0:
                line_num = line_num + 1
                num0 = num0 + 1
                label_str = str(line_num) + "\ttrain\t0\n"
                # label_list.append(label_str)
                fl.write(label_str)
                # data_str = func_str + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                # data_list.append(data_str)
                fw.write(data_str)
        else:
                line_num = line_num + 1
                num1 = num1 + 1
                label_str = str(line_num) + "\ttrain\t1\n"
                fl.write(label_str)
                # data_str = func_str + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                fw.write(data_str)  # 这样加载总的数据集的话，索引就是对应的位置
print("train数据个数：",line_num+1)
print("train0数据个数：",num0)
print("train1数据个数：",num1)
print(t)
with open(dataset+'_test.jsonl', 'rb') as ftest:
    for item in json_lines.reader(ftest):
        func_str = clean_str(item.get("func"))
        if item.get("target") == 0:
                line_num = line_num + 1
                label_str = str(line_num) + "\ttest\t0\n"
                # label_list.append(label_str)
                fl.write(label_str)
                # data_str = func_str + "\n"
                # data_str = ' '.join(func_str.split()[:400]) + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                # data_list.append(data_str)
                fw.write(data_str)
        else:
                line_num = line_num + 1
                label_str = str(line_num) + "\ttest\t1\n"
                fl.write(label_str)
                # data_str = func_str + "\n"
                # data_str = ' '.join(func_str.split()[:400]) + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                fw.write(data_str)  # 这样加载总的数据集的话，索引就是对应的位置

with open(dataset+'_valid.jsonl', 'rb') as fvalid:
    for item in json_lines.reader(fvalid):
        func_str = clean_str(item.get("func"))
        if item.get("target") == 0:
                line_num = line_num + 1
                label_str = str(line_num) + "\tvalidate\t0\n"
                # label_list.append(label_str)
                fl.write(label_str)
                # data_str = func_str + "\n"
                # data_str = ' '.join(func_str.split()[:400]) + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                # data_list.append(data_str)
                fw.write(data_str)
        else:
                line_num = line_num + 1
                label_str = str(line_num) + "\tvalidate\t1\n"
                fl.write(label_str)
                # data_str = func_str + "\n"
                data_str = ' '.join(func_str.split()[:1200]) + "\n"
                # data_str = ' '.join(func_str.split()[:400]) + "\n"
                fw.write(data_str)  # 这样加载总的数据集的话，索引就是对应的位置
fw.close()
fl.close()
