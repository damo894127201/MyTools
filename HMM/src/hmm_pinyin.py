#coding:utf-8

'''
基于HMM模型实现拼音输入法

@author Weiyang

edit_time : 2017-11-23

'''

class LanguageModel:
    '''
    语言模型，用于加载拼音到汉字的映射表，从语料库中生成：汉字之间的转移概率，汉字到拼音的发射概率，每个拼音下每个汉字的发生概率
    '''

    def __init__(self):
        '''
        #定义程序中间变量的数据结构
        '''

        #存放 拼音 映射到 汉字的 对照表
        self.state_value={} #格式为{拼音1：[汉字1，汉字2，汉字3,...],拼音2:[汉字1，汉字2,...],....}
        #存放 汉字 映射到 拼音的 生成概率 ，汉字到拼音的生成概率值我们默认同一个拼音下的所有汉字的概率值相等，即概率值为1/N ，N为某个拼音下的汉字个数
        self.emission_probability={} #格式为{汉字1：汉字1到拼音1的生成概率，汉字2：汉字2到拼音2的生成概率,...}
        #存放汉字1到汉字2的转移概率
        self.shift_probability={} #格式为{汉字1：{汉字1:概率值1，汉字2:概率值2，...},汉字2：{汉字1:概率值1，汉字2:概率值2，..},...}
        #存放拼音到汉字的概率，即社会用语中，同一个拼音下，不同汉字的使用频率值
        self.word_probability={} #格式为{汉字1:概率值1，汉字2：概率值2，汉字3：概率值3，...}
        #存放语料库，用于统计每个汉字的使用频率，以及汉字之间的转移概率
        self.corpus=[] #格式为：[汉字1，汉字2，汉字3，....]


    def ReadFile(self,pinyin_hanzi_path,corpus_path):
        '''
        #读取拼音到汉字的对照表 和 读取语料库
        :param pinyin_hanzi_path: 拼音到汉字的映射表的路径
        :param corpus_path: 语料库的路径
        '''
        #读取拼音到汉字的对照表
        with open(pinyin_hanzi_path,'r') as f:
            pass

        #读取语料库
        with open(corpus_path,'r') as f:
            pass



    def count_frequency(self):
        '''
        #统计语料库中，每个汉字的使用频率，以及汉字之间的转移概率,以及汉字到拼音的发射概率
        '''
        pass

    #初始化所需要的数据结构
    def init_datastructure(self):
        '''
        这里由于缺乏语料库和拼音汉字的映射表，因此便无法统计出，各个汉字之间的转移概率，即转移矩阵；
        汉字到拼音的发射概率，以及每个拼音下，单个汉字的使用频率；
        因此，我就自己构造了一个简单的转移矩阵，和发射概率矩阵 ；
        '''

        #初始化状态值列表，即初始化拼音到汉字的对照表
        self.state_value={'bei':['北','被','倍','杯'],'jing':['京','经','静','精'],'da':['大','打','达','搭'],'xue':['学','雪','穴','血']}
        #初始化汉字到拼音的发射概率
        self.emission_probability={'北':1.0/4.0,'被':1.0/4.0,'倍':1.0/4.0,'杯':1.0/4.0,
                                   '京':1.0/4.0,'经':1.0/4.0,'静':1.0/4.0,'精':1.0/4.0,
                                   '大':1.0/4.0,'打':1.0/4.0,'达':1.0/4.0,'搭':1.0/4.0,
                                   '学':1.0/4.0,'雪':1.0/4.0,'穴':1.0/4.0,'血':1.0/4.0}
        #存放汉字1到汉字2之间的转移概率
        self.shift_probability={'北':{'京':0.4,'经':0.2,'静':0.2,'精':0.2},'被':{'京':0.2,'经':0.4,'静':0.2,'精':0.2},
                                '倍':{'京':0.2,'经':0.2,'静':0.4,'精':0.2},'杯':{'京':0.2,'经':0.2,'静':0.2,'精':0.4},
                                '京':{'大':0.4,'打':0.2,'达':0.2,'搭':0.2},'经':{'大':0.2,'打':0.4,'达':0.2,'搭':0.2},
                                '静':{'大':0.2,'打':0.2,'达':0.4,'搭':0.2},'精':{'大':0.2,'打':0.2,'达':0.2,'搭':0.4},
                                '大':{'学':0.4,'雪':0.2,'穴':0.2,'血':0.2},'打':{'学':0.2,'雪':0.4,'穴':0.2,'血':0.2},
                                '达':{'学':0.2,'雪':0.2,'穴':0.4,'血':0.2},'搭':{'学':0.2,'雪':0.2,'穴':0.2,'血':0.4}}
        #存放同一个拼音下，不同汉字的使用频率
        self.word_probability={'北':0.4,'被':0.2,'倍':0.2,'杯':0.2,'京':0.4,'经':0.2,'静':0.2,'精':0.2,'大':0.4,'打':0.2,'达':0.2,'搭':0.2,
                               '学':0.4,'雪':0.2,'穴':0.2,'血':0.2}


class GraphNode:
    '''
    该类用于构造有向图的节点，该节点存储了每个汉字的，汉字实体，汉字到拼音的发射概率，汉字的前一个节点，以及最优路径时从起点到该节点的最大概率值分数

    我们以输入的拼音列表中的每一个拼音所对应的汉字集合里的每个汉字作为节点，构造一个有向图
    而每个节点之间的转移概率为边的权重，我们的目的便是要找到权重乘积最大的那条路径
    '''
    def __init__(self,word,emission):
        '''
        :param word: 是该节点所代表的汉字,即状态
        :param emission: 是该节点所代表汉字到拼音的发射概率，即状态值到观测值的生成概率
        '''

        self.word=word
        self.emission=emission

        #最优路径时，从起点到该节点的最大概率值分数
        self.max_score=0.0
        #最优路径时，该节点的前一个节点，这个变量用来输出路径(即输出汉字列表)的时候使用
        self.pre_node=None

class Graph:
    '''
    该类用于构造有向图
    '''

    def __init__(self,pinyin_list,loaded_data):
        '''
        根据拼音所对应的所有汉字组合，构造有向图
        :param pinyin_list: 输入的拼音列表
        :param loaded_data: 是类LanguageModel 的一个实例，目的用于加载拼音汉字的映射表，
                            汉字之间的转移概率，汉字到拼音的发射概率，在拼音给定下每个汉字的使用频率
        '''
        #将输入拼音列表和语言模型存为全局变量，便于后续函数的使用
        self.pinyin_list=pinyin_list
        self.loaded_data=loaded_data

        #根据拼音所对应的汉字组合，用于存储有向图
        self.sequence=[] #格式为：[{第一个拼音所对应的汉字GraphNode节点集合},{第二个拼音所对应的汉字GraphNode节点集合},{第三个拼音所对应的汉字GraphNode节点集合},,...]
        #即[{汉字1：汉字1的GraphNode,汉字2：汉字2的GraphNode节点,....},{汉字1：汉字1的GraphNode,汉字2：汉字2的GraphNode节点,....},...]

        #遍历输入的拼音列表，构造每个拼音下对应的汉字节点
        for pinyin in pinyin_list:
            #用于存放当前拼音对应的汉字节点，这里每个节点都是类GraphNode的一个实例
            current_position={} #格式为：{汉字1：汉字1的GraphNode,汉字2：汉字2的GraphNode节点,....}
            #从状态值列表Get_node_shift_emission_probability.state_value 中获取同一个拼音下，所有的汉字列表
            for word in self.loaded_data.state_value[pinyin]:
                #依据获取的汉字实体，从发射概率矩阵中获取汉字到拼音的发射概率
                emission=self.loaded_data.emission_probability[word]
                #有了汉字的实体，和其对应的发射概率，便可以构造当前拼音下的一系列图节点
                node=GraphNode(word,emission)
                #将该汉字节点加入当前拼音下的节点集合中
                current_position[word]=node
            #将该拼音下的节点集合加入到有向图中
            self.sequence.append(current_position)

        #用于存放各个汉字节点最大路径的概率得分，是一个字典
        self.viterbi_score={} #格式为：{0:[score1,score2,score3,..],1:[score1,score2,..],...}
        # 其中字典的键key为拼音列表中第k个拼音的下标，字典的值value是一个列表，里面存储的是相应拼音对应的汉字的得分数值
        #为便于使用存放各个汉字的概率得分，我们需要事先给它初始化
        for i in range(0,len(self.pinyin_list)):
            #当前拼音，可以用于获取该拼音下的汉字集大小
            current_pinyin=self.pinyin_list[i]
            #将每个拼音对应汉字的得分，初始化为None,大小为各个拼音对应的汉字集大小
            self.viterbi_score[i]=[None]*len(self.loaded_data.state_value[current_pinyin])


    def viterbi(self,position_t,hanzi_k):
        '''
        维特比算法，计算第t个拼音位置出现第k个汉字的概率，这里的t指的是输入拼音列表中的下标，
        k是指当前位置拼音所对应的汉字集合中的第k个汉字，注意下标都是从0开始的
        :param position_t: 输入拼音序列中的第 t 个拼音，同时也代表着图self.sequence中 第t个位置的 汉字节点集合
        :param hanzi_k: t 位置汉字节点集合中 第 k 个汉字
        :param loaded_data :用于加载转移概率，发射概率，拼音汉字映射表的语言模型LanguageModel实例
        :return: node.max_socre 返回从起点开始到当前汉字节点路径最大的概率值
        '''

        #先判断当前位置的汉字的最大概率是否已经计算
        if self.viterbi_score[position_t][hanzi_k] != None:
            return self.viterbi_score[position_t][hanzi_k]
        #当前position_t位置的拼音
        current_pinyin=self.pinyin_list[position_t]

        node=self.sequence[position_t][self.loaded_data.state_value[current_pinyin][hanzi_k]] #获取输入拼音中第t个位置的拼音的汉字集合中第k个汉字GraphNode节点
        if position_t ==0:
            #从语言模型LanguageModel中获得转移概率,由于是第一个拼音，这里便假定该拼音前面还有一个拼音，其发生概率为1，
            # 并且这里的转移概率我们使用各个汉字通常情况下的使用频率来确定，因为我们无法确定那个假定拼音所对应的汉字具体是什么，
            #这里获取第t个拼音位置为第k个汉字时，该汉字的使用频率的获取方式为：先获取t位置的拼音，之后以该拼音为字典的键key，
            # 从LanguageModel.state_value中获取该拼音所对应的汉字列表，然后从该列表中获取第 k 位置的汉字，
            # 进而以该汉字为字典的键key，从语言模型LanguageModel中获取汉字的使用频率
            hanzi=self.loaded_data.state_value[self.pinyin_list[position_t]][hanzi_k]
            state_shift=self.loaded_data.word_probability[hanzi]

            #获得该汉字对应的发射概率
            emission_probability=self.loaded_data.emission_probability[hanzi]

            #计算当前节点的概率得分
            node.max_score=1.0*state_shift*emission_probability
            #将当前节点的概率得分存入得分表self.viterbi_score中
            self.viterbi_score[position_t][hanzi_k]=node.max_score
            return node.max_score

        #获取当前汉字节点对应的汉字实体
        current_hanzi = self.loaded_data.state_value[self.pinyin_list[position_t]][hanzi_k]
        #获取前一个状态所有可能的汉字节点GraphNode集合
        pre_hanzi_list=self.sequence[position_t-1].keys() #self.sequence[position_t-1]返回的是一个字典，
        # 在这个字典中，键key为汉字实体，而值value为汉字对应的GraphNode节点,因此返回的是一个汉字列表
        pre_hanzi_list=list(pre_hanzi_list)

        # 用于记录前一个汉字的下标
        i = 0
        #对当前汉字节点，分别将其与前一个汉字节点的所有可能的汉字进行计算，我们要获取概率最大的那个组合，即pre_node ---current_node
        for pre_hanzi in pre_hanzi_list:

            #获取前一个汉字节点到当前汉字节点的转移概率
            state_shift=self.loaded_data.shift_probability[pre_hanzi][current_hanzi]

            #获取当前汉字current_hanzi对应的发射概率
            emission_probability = self.loaded_data.emission_probability[current_hanzi]

            #计算当前节点current_hanzi的概率得分
            score=self.viterbi(position_t-1,i)*state_shift*emission_probability

            #比较这个前驱节点到当前节点的得分值，是否为最大，我们需要保留得分最大的那个前驱节点
            if score>node.max_score :
                #存储比较大的得分
                node.max_score=score
                #存储相应的前驱节点为当前节点的前驱节点
                node.pre_node=self.sequence[position_t-1][pre_hanzi_list[i]]

            #下标增1，过渡到下一个前驱节点单词
            i+=1

        #将当前节点的最大得分存储到self.viterbi_score中
        self.viterbi_score[position_t][hanzi_k]=node.max_score
        return node.max_score

    #给定拼音输入列表，我们要输出概率最大的路径汉字序列
    def output_hanzi(self):
        #获取输入的拼音列表中最后一个位置的拼音
        end_pinyin=self.pinyin_list[-1]
        #用于存放当前位置拼音对应的各个汉字的最大概率
        hanzi_probability={} #格式为：{汉字1的索引:概率值1，汉字2的索引：概率值2，....}
        #遍历当前位置拼音所对应的汉字集合
        for hanzi_index in range(0,len(self.loaded_data.state_value[end_pinyin])):
            #计算当前位置拼音对应的第hanzi_index个汉字的最大概率值
            hanzi_probability[hanzi_index]=self.viterbi(len(self.pinyin_list)-1,hanzi_index)

        #从最后一个拼音对应汉字中选取概率最大的汉字所对应路径
        #我们可以通过获取汉字对应的索引，进而从self.sequence中获取该汉字对应的GraphNode节点,而该节点中存有相应的汉字，
        #再从GraphNode节点中获取其前驱节点
        #由于hanzi_probability是一个字典，因此，我们需要根据其值value来排序键key
        lst=sorted(hanzi_probability,key=hanzi_probability.__getitem__,reverse=True) #结果会返回一个逆序的列表，元素为相应的汉字索引
        max_pro_index=lst[0]
        #最后一个拼音对应的最大概率值的汉字
        max_pro_hanzi=self.loaded_data.state_value[self.pinyin_list[-1]][max_pro_index]


        #获取汉字
        hanzi_lst=[]

        #将最后一个拼音对应的概率值最大的汉字存入
        hanzi_lst.append(self.sequence[-1][max_pro_hanzi].word)
        pre_node=self.sequence[-1][max_pro_hanzi].pre_node #用于存放前驱节点

        #逆序遍历拼音列表，获取前驱节点对应的汉字
        while pre_node:
            hanzi_lst.append(pre_node.word)
            pre_node=pre_node.pre_node
        #输出拼音序列
        print('输入拼音是')
        print(self.pinyin_list)
        #此时获取的汉字是从后往前排列的，因此我们需要将hanzi_lst反转输出
        print('输出汉字是')
        print(hanzi_lst[::-1])



if __name__ == '__main__':
    #创建语言模型
    language_model=LanguageModel()
    language_model.init_datastructure()
    #构建图
    graph=Graph(['bei','jing','da','xue'],language_model)
    graph.output_hanzi()