# CMU_11785_Team_Project
Carnegie Mellon University 11785 Deep Learning team project

### 1. Data:
    数据文件夹，包含：
    1) 原始训练图片 train_imgs.tar 34184张150*150的3 channel彩图 
    2) 原始测试图片 test_imgs.tar 8197张150*150的3 channel彩图 （这里抱歉与之前说的5万张有出入，目前加起来一共4.2万多张）
    3) 文件列表 file_list  其中包含顺序/乱序的训练/测试数据列表，及对应标签。标签一共10类。
当初曾经将数据转换成lmdb格式，不过转换后文件较大（如训练数据为2.4GB），所以暂时不上传，后面如果需要可以直传。

### 2. Model：
    1) train_val_lvshq_128.prototxt 训练/测试模型文件，包含每层的结构。这里倒数第二层是128个neuron，相当于用128位（这里用的是二进制数）的数字来表示这张图像的特征。最后一层10个neuron对应10类标签。
    2) deploy_lvshq_128.prototxt 预测使用的模型文件，需要与训练相对应，只是去掉了最后一层。
    3) solver.prototxt 训练参数文件
    4) train.sh 训练脚本文件

### 3. Src：数据爬取和预处理的一些python和shell脚本。
    1) spider.py 数据爬取脚本，从jd.com和taobao.com爬取数据
    2) create_filelist*.sh 根据图片文件生成对应的图片列表文件，用于训练与测试。
    3) create_lmdb.sh 根据数据及标签生成对应的lmdb文件，用于训练与测试。
    4) create_meanfile.sh 根据Caffe的工具对数据做均值化（应该类似于feature zero mean）预处理
    5) convert_taobao_size.sh 把爬去的数据resize成想要的大小


注：
模型部分主要参考论文为：
1. Deep Learning of Binary Hash Codes for Fast Image Retrieval
Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

对应Github地址：https://github.com/kevinlin311tw/caffe-cvprw15



2. Supervised Learning of Semantics-Preserving Deep Hashing (SSDH)
Created by Kevin Lin, Huei-Fang Yang, and Chu-Song Chen at Academia Sinica, Taipei, Taiwan.

对应Github地址：https://github.com/kevinlin311tw/Caffe-DeepBinaryCode


