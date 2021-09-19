## 复现两篇GNN论文，分别是GCN和GAT  
## reproduce two GNN model, GCN and GAT  
## GitHub无法显示Latex公式，实验文章可以直接到我简书看[简书链接](https://www.jianshu.com/p/69f7e8a0a02c)  
使用的数据集：Citeseer, Cora, Pubmed
# GCN(2017)
其实GCN的原作者Kipf也自己发布了一个[Pytorch版本的GCN](https://github.com/tkipf/pygcn)，但我阅读了一下，感觉这个版本的GCN遗漏了太多细节。我按照GCN官方版本，复现了Pytorch。使用的是Citeseer、Cora和Pubmed三个数据集。accuracy和原论文的几乎无差异。
需要注意的几个细节如下：
* 每一层输入通道$H^{(l)}$必须先经过一层p=0.5的dropout
* L2 reg只作用在第一层的参数矩阵$W$
* 使用Xavier初始化
* 论文最后有个小实验，验证层数对结果的影响。文中只有第一层和最后一层有dropout。  
总的来说，我实现的版本不太稳定，换几个随机数种子，有一定的概率不收敛到相对接近原文的结果。但大多数情况下是能完美复现文中结果的。

实验结果如下（表格是准确率)：
|   层数  | Cora  | Citeseer  | Pubmed  |
|  ----  | ----  | ----  | ----  |
| 1  | 71.0% | 66.8%  | 72.4%  |
| 2  | 82.0% | 70.0%  | 79.1%  |
| 3  | 78.7% | 67.1%  | 77.7%  |
| 4  | 78.1% | 63.2%  | 76.8%  |
| 5  | 74.9% | 59.8%  | 69.7%  |
| 6  | 72.3% | 56.8%  | 68.9%  |
| 7  | 67.1% | 50.3%  | 68.8%  |
| 8  | 50.7% | 58.1%  | 71.5%  |
| 9  | 49.5% | 58.6%  | 52.0%  |
| 10  | 69.6% | 24.1%  | 67.8%  |

![epoch数-ACC](https://upload-images.jianshu.io/upload_images/21290480-64543b8f3f31bb7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![GCN层数-ACC](https://upload-images.jianshu.io/upload_images/21290480-98038575e710feef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# GAT(2018)
GAT模型的矩阵乘法比较繁琐复杂，因此不实现稀疏矩阵乘法。如果想要实现稀疏矩阵乘法， 可以使用PYG或者DGL作为框架。以下几个GitHub地址可以学习GAT模型：
[GAT官方实现 - TensorFlow](https://github.com/PetarV-/GAT)
[GAT - Pytorch实现，作为GAT框架](https://github.com/gordicaleksa/pytorch-GAT#training-gat)
[GAT - Pytorch论文复现](https://github.com/Diego999/pyGAT)
我发现了一个Paper和code有出入的地方。GAT模型里，邻接矩阵A的作用是作为attention mask。节点i的注意力除了i的所有邻节点，是否需要对自己进行注意力？如果存在自注意力，那邻接矩阵A需要变成A+I。文中的公式没有自注意力，但是代码里面有。

我的GAT在cora的复现结果只有81.%，达不到GAT论文里的83%。Citeseer也是，validation set的ACC可以冲到73%，但是test set的ACC不到71%，并不显著高于GCN的ACC。我也不知道为什么，如果有人知道的话可以告诉我。实验结果如下（表格是准确率)：
| Cora  | Citeseer  |
| ----  | ----  |
| 81.1% | 70.8%  |  

![GAT_epoch_ACC](https://upload-images.jianshu.io/upload_images/21290480-5deff7e5d95dbdb9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# Simple Graph Convolution(2019)
接下来我要对GCN做进一步分析。第一层的ReLU函数是否对半监督分类有显著的影响？我试验了带有ReLU（也就是论文原版模型），以及去掉ReLU函数的结果。ACC运行5次取平均：
|     | Cora  | Citeseer  | Pubmed  |
|  ----  | ----  | ----  | ----  |
| 带有ReLU | 81.28% | 71.20%  | 79.38%  |
| 不带ReLU | 81.18% | 71.28%  | 79.26%  |  

结论很明显，是否带有ReLU函数对结果没有影响。
如果一个K层GCN的中间都没有激活函数，或者说中间都是$\sigma (x)=x$这样的线性激活函数，那么GCN就变成了：$$\widehat{y}=softmax(S^{K}X\Theta _1\Theta _2...\Theta _K)$$直接用一个参数矩阵代替K个参数矩阵，GCN就变成了：$$\widehat{y}=softmax(S^{K}X\Theta )$$其中$$S=(D+I_N)^{-\frac{1}{2}}(A+I_N)(D+I_N)^{-\frac{1}{2}}$$
$S^{K}X$是可以在模型训练前预先计算出来储存到内存里的。这个时候我们就得到了Wu et al. (2019)提出的Simple Graph Convolution。Wu在文中证明了此GCN是Graph上的低通滤波器。如果K的数量越大，会有越多图信号被过滤掉，并且随着层数增加，图信号被过滤的数量收敛到100%，这很好地解释了为什么GCN做不深。文中实验结果显示，简化版的GCN分类效果并不比原版GCN差，并且速度快了两个数量级。
# 参考文献
[1] Kip F  T N ,  Welling M . Semi-Supervised Classification with Graph Convolutional Networks[J].  2016.  
[2] Velikovi P ,  Cucurull G ,  Casanova A , et al. Graph Attention Networks[J].  2017.  
[3] Wu F ,  Zhang T ,  Souza A , et al. Simplifying Graph Convolutional Networks[J].  2019.