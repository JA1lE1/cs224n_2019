# cs224n word2vec Homework

[TOC]

## 代码过程

### distinct——words

- python	

  - list.extend()

    - 区别和list.append的区别

  - comprehensionway

    ```python
    ##the loop way
    #The list of lists
    list_of_lists = [range(4), range(7)]
    flattened_list = []
    
    #flatten the lis
    for x in list_of_lists:
        for y in x:
            flattened_list.append(y)
            
            
    #List comprehension way
    #The list of lists
    list_of_lists = [range(4), range(7)]
    
    #flatten the lists
    flattened_list = [y for x in list_of_lists for y in x]
    ```

  - set

    - set（list）可以剔除list中的共有的元素

- [this](https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python)

  - 这是loop的python的缩写，有点6

  - set-to remove duplicate words
    - what is difference between list and set?





### copute_co_ooccuttence_matrix

- 有
  - corpus
    - 两句话
    - 两句话的每个独立的单词组成的list，没有重复
    - 这个list的长度
  - window_size
- M是一个矩阵
  - n*n,n =num_words
  - 遇到到的效果
    - 是找到list中每一个单词前w_size个和后w_size个
    - 注意到对角线全是0，先填充
    - eg，下标0,1怎么填写。找到下表0的单词，然后在对应的corpus的句子中找到对应的位置，取前4个和后4个，对每个对应的单词找到其在list中的下标记n，然后对（0，n）那个格子+1
    - 估计复杂度
      - 扫描n次，在n次中扫描至多w_size*2次，其中n为list中每个value的len
        - 进而想，如果一篇语料库有100万个单词，看起来也还好
- word2Ind
  - 是字典
  - 键就是字，value就是从M中copy
- 



- 疑问
  - numpy中的数据能是复合类型的吗？
    - 例如又有char，又有int
  - python
    - 字典中可以用for吗

- 参考

  - [github上某大佬代码](https://github.com/Luvata/CS224N-2019/blob/master/Assignment/a1/cs224n-2019-as1.ipynb)

    - 完全理解word2Ind是语料库单词在排序单词表中的排序(这个排序表有函数distinct_words生成)

    - ```python
      indices = [word2Ind[i] for i in sentence]
      ```

      这个是有点秀的

      思路清晰- 需要得到下表-在排序表的下标记即word2Ind[i]，而i就是单词，而单词在哪里，在sentence中。即sentence in corpus

      ```python
      while current_index < sentence_len:
          left  = max(current_index - window_size, 0)
          right = min(current_index + window_size + 1, sentence_len) 
          current_word = sentence[current_index]
          current_word_index = word2Ind[current_word]
          words_around = indices[left:current_index] + indices[current_index+1:right]
      
          for ind in words_around:
              M[current_word_index, ind] += 1
      ```

      而这一段的思路就是，需要写的是M的内容。而M的内容是由下标决定，第一下标是sentence中扫描的word的下标，（这里这一段代码就显得不是非常的完美），而第二下标就是当前被扫描到的word的nearby的word的下标，（nearby的尺度由window_size决定），而这里用到的一个思路是以目的分割代码表示，就是words_aroud 的 + 的表现意义，空间上这段代码自然比较小，它只存储了所需改变的下标，我的代码三层循环，大佬的代码两层，我多出来的是扫描word的nearby的过程，大佬是通过计算替代循环，==待续==



### reduce_to_k_dim

- this quiz inplement the sklearn 
  - sklearn 的通用形式fit:point_right: transform????
  - 一般有fit transform fit_transform ??



- 出现的问题
  - 理论知识的缺乏
- 参考（暂未）
  - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD.fit)
  - [sklearn_cookbook](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781783989485/1/ch01lvl1sec21/using-truncated-svd-to-reduce-dimensionality)
  - [svd](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/static/pdf/web/viewer.html?file=https%3A%2F%2Fdavetang.org%2Ffile%2FSingular_Value_Decomposition_Tutorial.pdf)





### plot_embeddings

- 出现的问题
  - 貌似notebook的编译有问题 没有自动缩进

  - 可以因为在环境中装了一些东西又瞎卸载导致环境出了很大的问题

  - 这里出现的问题是真滴皮（多）

    - 首先必须感谢[这个大佬的github代码](https://github.com/Luvata/CS224N-2019/blob/master/Assignment/a1/cs224n-2019-as1.ipynb)

    - 细节

      - 首先好几个地方我都犯了自动生成变量未检查导致变量参量和实际变量的不同而照成对

        不同的数据处理时的bug

      - 在def compute_co_occurrence_matrix(corpus, window_size=4):中我对num_words , 使用成了

        上一个cell出现的类似变量，要知道在新的def中是大忌

      - ==最简单的检查就是对新定义的def函数的每个参量进行检查==

      - ==然后就是对每个参量的意义的理解，即每个代码的伪代码的实现对应==

- 参考







### Co-Occurrence Plot Analysis 

- 参考
  - [numpy.linalg.norm](https://blog.csdn.net/lanchunhui/article/details/51004387)
    - 就是计算范数
  - [newaxis](https://blog.csdn.net/lanchunhui/article/details/51004387)
    - 放在哪就在在哪增加一个维度

