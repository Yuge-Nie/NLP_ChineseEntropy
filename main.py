# coding=gbk
import jieba
import os
import re
import math


# 读取文件夹文件
def read_data(path):
    data_txt = []
    files = os.listdir(path)  # 返回指定的文件夹包含的文件列表
    for file in files:
        position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符

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
        position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符
        with open(position, "r", encoding='ANSI') as f:
            for lines in f.readlines():
                with open("C:\\Users\\NYG\\PycharmProjects\\pythonProject\\text2\\all.txt", "a", encoding='ANSI') as p:
                    p.write(lines)

# 分字
def cut_word(path):
    with open(path, 'r', encoding='ANSI') as file:
        cs = {}
        for line in file:
            line = line.strip()
            for c in line:
                if is_chinese(c):
                    if cs.get(c):
                        cs[c] += 1
                    else:
                        cs[c] = 1
                else:
                    continue
        for w in cs.keys():
            pw = cs[w]/len(cs)
            cs[w] = pw
    ## 写字频
    # with open('.\\text1\\cs.txt', 'w', encoding='utf-8') as csf:
    #     for c in cs.keys():
    #         csf.write(c + ':' + str(cs[c]) + '\n')
    # # 写字概率
    # with open('.\\text1\\p_cs.txt', 'w', encoding='utf-8') as csf:
    #     for w in cs.keys():
    #         csf.write(w + ':' + str(cs[w]) + '\n')
    # print('字概率保存成功')
    # return cs
# 分词
def cut_vocab(path):
    with open(path, 'r', encoding='ANSI') as file:
        vocab = {}
        sentences = cut_sentences(file.read())
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip()
            cut = jieba.lcut(sentences[i])
            for j in range(len(cut)):
                if is_chinese(cut[j]):
                    if vocab.get(cut[j]):
                        vocab[cut[j]] += 1
                    else:
                        vocab[cut[j]] = 1
                else:
                    continue
        for w in vocab.keys():
            pw = vocab[w] / len(vocab)
            vocab[w] = pw
    ## 写词频
    # with open('.\\text1\\vocab.txt', 'w', encoding='utf-8') as vf:
    #     for w in vocab.keys():
    #         vf.write(w + ':' + str(vocab[w]) + '\n')
    # print('词频保存成功')
    # # 写词概率
    # with open('.\\text1\\p_vocab.txt', 'w', encoding='utf-8') as vf:
    #     for w in vocab.keys():
    #         vf.write(w + ':' + str(vocab[w]) + '\n')
    # print('词概率保存成功')

    return vocab
# 分双字
def cut_bics(path):
    with open(path, 'r', encoding='ANSI') as file:
        bics = {}
        sentences = cut_sentences(file.read())
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
    ## 写词频
    # with open('.\\text1\\vocab.txt', 'w', encoding='utf-8') as vf:
    #     for w in vocab.keys():
    #         vf.write(w + ':' + str(vocab[w]) + '\n')
    # print('词频保存成功')
    ## 写词概率
    # with open('.\\text1\\p_bics.txt', 'w', encoding='utf-8') as vf:
    #     for w in bics.keys():
    #         vf.write(w + ':' + str(bics[w]) + '\n')
    # print('双字概率保存成功')

    return bics
# 分双词
def cut_bivocab(path):
    with open(path, 'r', encoding='ANSI') as file:
        bivocab = {}
        sentences = cut_sentences(file.read())
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
    ## 写词频
    # with open('.\\text1\\vocab.txt', 'w', encoding='utf-8') as vf:
    #     for w in vocab.keys():
    #         vf.write(w + ':' + str(vocab[w]) + '\n')
    # print('词频保存成功')
    ## 写词概率
    # with open('.\\text1\\p_bivocab.txt', 'w', encoding='utf-8') as vf:
    #     for w in bivocab.keys():
    #         vf.write(w + ':' + str(bivocab[w]) + '\n')
    # print('双词概率保存成功')

    return bivocab

# 算熵
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

# 判断是否是中文
def is_chinese(str):
    for i in str:
        if i >= '\u4e00' and i <= '\u9fa5':
            flag = True
        else:
            return False
    return flag
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


if __name__ == '__main__':
    # preprocess()
    cut_vocab('.\\text1\\all.txt')
    cut_bivocab('.\\text1\\all.txt')
    cut_word('.\\text1\\all.txt')
    cut_bics('.\\text1\\all.txt')
    # vocab = cut_vocab('.\\text1\\all.txt')
    # bivocab = cut_bivocab('.\\text1\\all.txt')
    # aver_entropy = cal_entropy('.\\text2\\all.txt', vocab, bivocab)
    # cs = cut_word('.\\text1\\all.txt')
    # bics = cut_bics('.\\text1\\all.txt')
    # aver_entropy = cal_entropy('.\\text2\\all.txt', cs, bics)
    # print(aver_entropy)
    print('finish')
