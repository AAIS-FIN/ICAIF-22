# 本次汇报的论文
- 3.Multivariate Realized Volatility Forecasting with Graph Neural Network
- 19.Synthetic Data Augmentation for Deep Reinforcement Learning in Financial Trading
- 34.Asset Price and Direction Prediction via Deep 2D Transformer and Convolutional Neural Networks
- 39.Temporal Bipartite Graph Neural Networks for Bond Prediction
- 47.Objective Driven Portfolio Construction Using Reinforcement Learning

# 3.Multivariate Realized Volatility Forecasting with Graph Neural Network
## 论文简述
本文提出一种多变量预测方法，通过限价订单簿数据LOB来预测股票市场的短期波动。具体而言，本文通过加入GTN（Graph Neural Networks）图神经网络的方法（可以捕捉时序股票之间复杂关系，不仅仅是股票时序的协方差矩阵，还可以聚合节点之间的信息），以股票的种类和时间作为图的节点，通过训练目标指标、GICS行业细分和行业供应链之间的关系连接不同节点之间的边，从而实现数据的图结构化，提出的模型简称为`GTN-VF`。最后，本文以标普500的已实现波动率进行实证，模型结果相比于基线模型`Naive Guess`、`HAR-RV`，`LightGBM`，`MLP`，`TabNet`在不同$\Delta T$下有更小的RMSPE，证实了`GTN-VF`方法的有效性和稳健性。同时，本文提出的`GTN-VF`模型在考虑资产之间关系的情况下，比基准模型以及单独考虑关系信息的模型在所有预测时段上都表现出色。

## 论文核心方法
![](https://obsidian-1314863343.cos.ap-beijing.myqcloud.com/obsidian/pic/20230911142231.png)

本文本质上是一个回归预测问题，希望套用图神经网络框架获得比基线模型更好的预测结果，核心是如何设计神经网络的图结构。本文图结构点用的是不同股票的时间状态，边用的是目标近似的Top-K，行业之间的关系（不同粒度细分）和供应链之间的关系来构建边。而选取GTN框架的原因是有研究表明利用transformer模型可以获得更好的训练结果。

## 对论文的思考
- 模型解释性：虽然该论文利用GTN在预测短期波动性方面表现优秀，但对于模型的解释性较为欠缺。未来的改进可以探索如何更好地解释模型的预测结果，例如通过可解释的图结构或特征的重要性分析，或者加入有关金融先验知识（如波动率曲面）。
- 不同市场的验证：论文实验结果在S&P 500指数的500只股票上的表现优于其他基准模型。为了提高对于该模型的说服力，可以在多个数据集或具有不同特征的市场上进行进一步的验证，比如加入中国市场的ETF50。
- 模型结构：本文是采用end to end学习方式，但是用前$\Delta T^ \prime$的数据预测后$\Delta T$的数据，应该是seq to seq的学习方式更合适，可以在整体上加入类似于RNN的模型。 

# 19.Synthetic Data Augmentation for Deep Reinforcement Learning in Financial Trading
## 论文简述
本文主要研究问题是通过使用增强的合成数据（由TimeGAN实现）来训练深度强化学习（DRL）的agent进行金融交易，以解决金融交易中的数据稀缺性问题。通过使用真实和合成数据集进行训练，观察训练过程中奖励的稳定性，并评估其在股票市场和加密货币市场（实证采用的是谷歌股票和比特币）上的性能，得到结论：由合成数据训练的agent比由真实数据训练的agent获得的利润更高，同时具有相似的鲁棒性。

## 论文核心方法
前人TimeGAN方法简述：
- TimeGAN认为时间序列含有两种特征：static features 和temporal features，于是定义了两个目标函数：全局Jensen-Shannon散度和局部Kullback-Leibler散度。
- TimeGAN包括四个部分：embedded function，recovery function，sequence generator和sequence discriminator。前两个是自编码的encoder和decoder，后两个是GAN的生成器和判别器，因此TimeGAN同时学习嵌入特征，生成表示，并随时间迭代。嵌入网络提供了潜在的空间表示，对抗网络在该空间内判别，真实数据和合成数据的潜在编码通过监督损失而同步。

本文方法的创新在于利用GAN的方法，通过模拟真实数据，解决了强化学习中数据稀缺的问题。

1.在GAN方面，本文采用TimeGAN结构，能够保留原始数据集中的高时间相关性和空间关系。具体而言，首先采用autoencoder（AE）重建时间序列，再采用GRU的自监督学习，最后使用真实和随机噪音联合训练GAN的生成器和判别器。

2.在DRL方面，具体而言，本文基于FinRL框架，在单一资产交易问题上（假设交易过程是MDP，资产价格不会因买卖而发生浮动，并考虑手续费），采用滚动窗口的训练方式，分别采用A2C和PPO的深度强化学习算法，设置`State Space 𝒔 = [P, V, SMA, RSI, OBV, M, Position]`，`Action Space 𝒂`为buy、sell和hold整数量的资产，制定好`Policy 𝜋 (𝑠)`，采用T+1策略，希望最大化净资产为目标，进行Adam优化训练。

## 对论文的思考
- 本论文研究的是单一资产和单一智能体，可以进阶考虑多智能体（multi-agent）与多资产投资组合的情形，寻找多智能体的帕累托最优解。
- 本文谷歌股票的train和test之比是15:1，训练网络数据过大，而且是模拟模块和强化学习模块是分开进行的，可以考虑边模拟边学习的方式抓住应对近期股票市场涨跌的策略，以减小训练。

# 34.Asset Price and Direction Prediction via Deep 2D Transformer and Convolutional Neural Networks
## 论文简述
本文的主要研究问题是如何使用CV技术预测金融资产价格并开发对应算法交易策略（采用CNN的类似研究也不多）。本文开发了两种方法，分别基于二维深度注意力神经网络(DAPP)和基于二维深度补丁嵌入卷积神经网络(DPPP)。文章通过分析不同金融时间序列ETF数据，通过OHLCV以及Ta-Lib上的技术指标，将其转换为二维图像（$65 \times 65$）作为输入，以buy，sell，hold作为分类输出，从而实现择时策略。本文实验结果表明，DAPP和DPPP的预测准确度较高，并且在长期的样本外测试期内表现优于基线方法和买入持有策略。此外，注意力机制和补丁嵌入能够提高资产价格和方向预测性能。

## 论文核心方法
本文主要思路是将金融资产的时间序列数据转换为二维图像，最后接一个分类器作为buy，sell，hold的交易策略。核心部分是图像处理的两种特征方法：DAPP方法使用注意力机制增强模型对关键信息的关注，而DPPP方法则使用补丁嵌入结构来捕捉图像中的局部特征。

1.对于DAPP：DAPP的一个挑战是模型的运行时间与像素数量成二次方，Vision Transformer 通过将图像分成块并分别对每个块应用自注意力来解决这个问题。文章还采用了清晰度感知最小化（SAM）技术来提高图像质量并使其更加稳健。

2.对于DPPP：ConvMixer 架构认为 Vision Transformer 成功的主要原因可能是在输入表示中使用patch，而不是 Transformer 架构（即自注意力机制）。所以DPPP实际上是一个对照实验，其目标用卷积层是复制self-attention的效果。

## 对论文的思考
- 文章提到数据量较少，使得DAPP无法充分发挥其最佳性能，没有与DPPP对照模型很好的区分开，可以考虑加入GAN进行数据模拟。
- 在不平衡数据集里面，模型很难区分买入和卖出的图像和持有的图像，导致模型大多数情况下预测大多数图像为持有状态，这说明简单的`Buy and Hold`的长期保守投资策略在该数据集下是不起效的，需要进一步指定具体买卖的数量，或者将数量作为分类。

# 39.Temporal Bipartite Graph Neural Networks for Bond Prediction
## 论文简述
本文主要研究问题是如何利用债券基金的数据信息来预测债券价格和收益率。由于二级市场中债券交易不频繁，导致数据观测不连续和存在缺失。为了解决这一挑战，本文并不采用传统的缺失值插补，而是建立一种基于时间双分图神经网络（TBGNN）的模型，包括学习债券和基金（债券基金）之间的节点嵌入以及它们相关因素的双分图表示模块，用于建模时间间隔的LSTM模块，以及使用图结构对未标记节点表示进行正则化的自监督目标。通过小批量随机梯度下降的训练过程，减轻了不同模块和目标的模型复杂性和计算成本。研究结果表明，TBGNN模型在债券价格和收益率预测上提供了更准确的预测能力（评价标准为$RSME,MAE,MAPE,R^2$，对照的基线模型为LSTM、GRU、GCN-LSTM、GraphSage-LSTM和GAT-LSTM）。

## 论文核心方法
![](https://obsidian-1314863343.cos.ap-beijing.myqcloud.com/obsidian/pic/20230915174747.png)

本文的核心思路是通过将多个债券和多个基金的数据设计为二分图的信息传递形式（也可理解为共同的债券持有人进行信息传递），将基金包含的债券进行连边，债券权重比例作为边的权重，并对节点进行图嵌入。

![](https://obsidian-1314863343.cos.ap-beijing.myqcloud.com/obsidian/pic/20230915174449.png)

其次，对债券输出特征输入到LSTM来学习每个时间点之间的时间依赖性，最终通过MLP得到债券收益率。所有 lag-p 图共享相同的二分图神经网络。

最后，为解决**supervision-starvation**问题，希望得到一个更合理的图嵌入特征，本文定义一个**self-supervision**模块，通过加入节点嵌入对权重矩阵的拟合损失作为原损失函数的惩罚项，充分解释基金的投资金额对债券价格影响。

除与基线模型对比预测性能外，为了提高模型的可解释性并评估自变量对预测值的影响，本文还设计了在预测债券价格和收益率时包含/排除债券特征的实验。

## 对论文的思考
- 本文并没有直接解决二级市场交易不频繁带来的数据缺失问题，可以考虑采用如GAN的方法模拟空缺数据的价格走势，再用TBGNN与基线模型对比模型效果会更有说服力。
- 本文方法的主要图模型是债券和基金的二分图，属于异构图的一种。由于债券基金与股票市场息息相关（二级债券基金通过参与二级市场股票投资，增厚基金收益），可以考虑继续加入关于股票的异构节点，将其与债券基金节点连边，从而包含更多信息，弥补连续时间上缺失数据的预测信息（股票市场相对活跃，包含更多价格信息）。

# 47.Objective Driven Portfolio Construction Using Reinforcement Learning
## 论文简述
本文的主要研究问题是利用深度学习和机器学习方法建模交易者行为，具体方法是加入3个投资者目标：Information Ratio，Maximum Drawdown和Turnover，实证以BIST 100指数进行分析。最后本文还希望对模型进行解释，采用传统方法和简单机器学习模型来模拟神经网络结果（包括Lasso，Elastic Net和Random Forest），依据$R^2$来解释，同时还采用Random Forest自带的特征重要程度排序，验证了传统线性方法的不足，并探究了不同超参数下DRL方法的表现。

## 论文核心方法
首先本文基于已有的前人强化学习投资框架**AlphaPortfolio**如下：
![](https://obsidian-1314863343.cos.ap-beijing.myqcloud.com/obsidian/pic/20230917092411.png)
框架模块解释：
- 最新的序列表示提取模型（sequence representation extraction models， SREM），如Transformer编码器（TE）和长短期记忆（LSTM），以便灵活有效地表示和提取来自输入特征的时间序列的信息，如公司的基本面和市场信号，即环境状态。
- 跨资产注意力网络（cross-asset attention networks， CAANs），可捕捉跨资产的属性互动，本质上是利用注意力机制的Q，K，V矩阵计算不同资产之间的相关程度，最后生成一个 "赢家得分"，对资产和交易（policy和action）进行排名，随后评估投资组合的表现，即考察回报。
- 接下来，**AlphaPortfolio**构建了一个多空组合，在赢家得分高的资产中持有多头，在赢家得分低的资产中持有空头（也确定了股票池）。
- 最后采用强化学习优化学习投资组合权重（action）。

本文主要方法是通过强化学习来获取更好的投资策略，增加了三个投资者目标（也是本文的创新点）：(1) 通过最大化信息比率来实现超额alpha；(2) 通过优化最大回撤调整回报来降低下跌风险；(3) 通过限制换手率来降低交易成本。具体伪代码如下：
![|500](https://obsidian-1314863343.cos.ap-beijing.myqcloud.com/obsidian/pic/20230916135006.png)

## 对论文的思考
- 本文采用随机森林解释特征重要性，并证实传统线性模型的劣势，只是说明深度神经网络的拟合预测优势，但是强化学习的策略模拟行为并没有得到很好的解释，而且选股策略本质上还是人提出的多空策略，强化学习只是进行配股。
- 此外，本文提出的目标驱使强化学习仍是单一目标（伪代码中多目标是if-else结构），但为更好的模拟投资人，应该同时考虑多目标，可以考虑加入多目标强化学习模型（比如hard-share结构）。

# 未来可研究的方向
- 目前深度强化学习方法DRL已经成为研究投资组合方法论的热门方法，主要关注于金融资产的时间序列数据，但是结合市场情绪分析和金融文本分析的RL方法还寥寥无几。因此，我们可以考虑将市场情绪和文本也作RL中environment的一部分，或者多加一个应对市场情绪和文本的agent形成多智能体问题，使得投资决策更加智能和稳健。
- 图神经网络对于股票投资组合的应用也较少，大多数都将GNN用于信用评级上。但是图神经网络对应的推荐算法应用可以很好的应用在制定股票池上，也就是选股。投资组合的权重可以由DRL迭代学习具体配股，而择时可以用DRL或者某种神经网络结构上层添加一个分类器。目前暂且没有看到选股、配股（对冲）、择时一体化的深度学习算法，但是DRL框架可以做到。
- 如何fine-tune神经网络的超参数是一个具有挑战的问题，目前借助meta-learning对金融深度模型结构调参的研究极少，但这是具有应用前景的，我们可以考虑比如DRL中action种类数设置，GNN中的行业细分粒度，VIT中图像patch的维度等调参或调网络结构的问题。
- 对于金融资产的时间序列预测问题上，一些文章认为GAN的模拟方法得到的效果比传统深度学习方法如LSTM好，但生成器学习迭代训练低效，为减轻计算量，可以输入其他相关时间序列或文本的前后语义信息，得到高效且逼真的数据模拟。
