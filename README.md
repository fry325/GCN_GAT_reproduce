 复现两篇GNN论文，分别是GCN和GAT  
 reproduce two famous GNN model, GCN and GAT  
复现实验结果在我的简书博客里[复现结果链接](https://www.jianshu.com/p/69f7e8a0a02c)  
使用的数据集：Citeseer, Cora, Pubmed  
模型核心代码：GCN.py和GAT. py
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
