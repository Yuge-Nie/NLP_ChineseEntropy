# NLP_Entropy-of-Chinese
# 中文分词

### 1 任务需求

​       分别以词和字为单位计算数据库中的小说的平均信息熵，熟悉自然语言处理的方法与过程。

### 2 任务知识

​		根据Peter F.Brown和Stephen A.Della Pietra的论文《An Estimate of an Upper Bound for the Entropy of English》
### 3 任务过程

#### 3.1 数据预处理		

​		首先，将数据文本分为两份，一份用来训练模型，一份用来计算平均信息熵。对数据进行预处理，将所有文本整合到同一个txt文件中。

```python
# 读取文件夹文件
def read_data(path):
    data_txt = []
    files = os.listdir(path)  # 返回指定的文件夹包含的文件列表
    for file in files:
        position = path + '\\' + file
        with open(position, 'r', encoding='ANSI') as f:
            data = f.read()
            data_txt.append(data)
        f.close()
    return data_txt, files
# 模型语料库预处理
def preprocess():
    path = "C:\\Users\\NYG\\PycharmProjects\\pythonProject\\text2"
    data_txt, filenames = read_data(path)
    for file in filenames:  # 遍历文件夹
        position = path + '\\' + file
        with open(position, "r", encoding='ANSI') as f:
            for lines in f.readlines():
                with open("C:\\Users\\NYG\\PycharmProjects\\pythonProject\\text2\\all.txt", "a", encoding='ANSI') as p:
                    p.write(lines)
```

#### 3.2 训练模型

​		然后，对训练集使用jieba进行分词和分字，分别计算他们作为单个词/字出现的概率，即用频率除以总数。

```python
def cut_vocab(path):
    with open(path, 'r', encoding='ANSI') as file:
        vocab = {}
        sentences = cut_sentences(file.read())	#分句
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip()	#去除掉空格
            cut = jieba.lcut(sentences[i])	#使用jieba分词
            for j in range(len(cut)):
                if is_chinese(cut[j]):	#去除掉符号
                    if vocab.get(cut[j]):
                        vocab[cut[j]] += 1
                    else:
                        vocab[cut[j]] = 1
                else:
                    continue
        for w in vocab.keys():
            pw = vocab[w] / len(vocab)
            vocab[w] = pw
    return vocab
# 分字
def cut_word(path):
    with open(path, 'r', encoding='ANSI') as file:
        cs = {}
        for line in file:
            line = line.strip()	#去除掉空格
            for c in line:
                if is_chinese(c):	#去除掉符号
                    if cs.get(c):
                        cs[c] += 1
                    else:
                        cs[c] = 1
                else:
                    continue
        for w in cs.keys():
            pw = cs[w]/len(cs)
            cs[w] = pw
    return cs
```

​		在此任务中使用bi-gram模型,因此将连在一起使用的两个词/字也分割出来，并计算他们的概率。

```python
# 分双词
def cut_bivocab(path):
    with open(path, 'r', encoding='ANSI') as file:
        bivocab = {}
        sentences = cut_sentences(file.read()) #分句
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip()
            cut = jieba.lcut(sentences[i])
            for j in range(len(cut) - 1):
                if is_chinese(cut[j]):
                    for k in range(j + 1, len(cut)):
                        if is_chinese(cut[k]):
                            if bivocab.get(cut[j] + ' ' + cut[k]):
                                bivocab[cut[j] + ' ' + cut[k]] += 1
                                j += 1
                                k += 1
                            else:
                                bivocab[cut[j] + ' ' + cut[k]] = 1
                                j += 1
                                k += 1
                        else:
                            continue
                else:
                    j += 1
        for w in bivocab.keys():
            pw = bivocab[w] / len(bivocab)
            bivocab[w] = pw
    return bivocab
# 分双字
def cut_bics(path):
    with open(path, 'r', encoding='ANSI') as file:
        bics = {}
        sentences = cut_sentences(file.read()) #分句
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip()
            for j in range(len(sentences[i]) - 1):
                if is_chinese(sentences[i][j]):
                    for k in range(j + 1, len(sentences[i])):
                        if is_chinese(sentences[i][k]):
                            if bics.get(sentences[i][j] + ' ' + sentences[i][k]):
                                bics[sentences[i][j] + ' ' + sentences[i][k]] += 1
                                j += 1
                                k += 1
                            else:
                                bics[sentences[i][j] + ' ' + sentences[i][k]] = 1
                                j += 1
                                k += 1
                        else:
                            continue
                else:
                    j += 1
        for w in bics.keys():
            pw = bics[w] / len(bics)
            bics[w] = pw
    return bics
```

在分词、分双词和双字过程中是先对文本分句，再进行分词的，这样可以保证不会和下一句的内容联系到一起。并且会检查是否是中文，这样可以清除掉各种标点符号。

```python
# 分句
def cut_sentences(content):
    end_flag = ['?', '!', '.', '？', '！', '。', '…', '......', '……', '\n']
    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        tmp_char += char
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break
        if char in end_flag:
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''
    return sentences
# 判断是否是中文
def is_chinese(str):
    for i in str:
        if i >= '\u4e00' and i <= '\u9fa5':
            flag = True
        else:
            return False
    return flag
```

#### 3.3 计算平均信息熵

​	最后就是对另一个文本进行词/字的平均信息熵计算，之前算好的词/字和双词/双字的概率存在字典中，通过调用算好的概率，用公式$(6)$来计算测试文本的值，这里也是先计算每一句的entropy，最后用所有句子的entropy总和除以句子总数。

```python
def cal_entropy(path, v, biv):
    with open(path, 'r', encoding='ANSI') as file:
        sentences = cut_sentences(file.read())
        entropy_sum = 0
        for i in range(len(sentences)):
            entropy = 0
            sentences[i] = sentences[i].strip()
            t = jieba.lcut(sentences[i])
            flag = 1
            for j in range(len(t) - 1):
                if is_chinese(t[j]):
                    if flag == 1:
                        if t[j] in v:
                            entropy = v[t[j]]
                        else:
                            entropy = 1 / len(v)
                        flag = 0
                    for k in range(j + 1, len(t)):
                        if is_chinese(t[k]):
                            if t[j] + ' ' + t[k] in biv:
                                entropy *= biv[t[j] + ' ' + t[k]]
                            else:
                                entropy *= 1/len(biv)
                            j += 1
                            k += 1
                        else:
                            continue
                else:
                    j += 1
            if entropy >= 0.00000000000001e-128:
                entropy_sum += (-1) / len(t) * math.log(entropy, 2)
    return entropy_sum / len(sentences)

if __name__ == '__main__':
    ## 预处理
    # preprocess()
    ## 算词
    # vocab = cut_vocab('.\\text1\\all.txt')
    # bivocab = cut_bivocab('.\\text1\\all.txt')
    # aver_entropy = cal_entropy('.\\text2\\all.txt', vocab, bivocab)
    # 算字
    cs = cut_word('.\\text1\\all.txt')
    bics = cut_bics('.\\text1\\all.txt')
    aver_entropy = cal_entropy('.\\text2\\all.txt', cs, bics)
    print(aver_entropy)
```

### 4 任务结果

通过计算以词为单位的平均信息熵为：

```python
6.4982451458436135
```

以字为单位的平均信息熵为：

```python
6.463055672802491
```



