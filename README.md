# FedHQ (PyTorch)

Implementation of the vanilla federated learning paper : [FEDHQ: Dynamic Aggregation for HeterogeneousQuantization in Federated Learning]().

## References
Our experiements refer to the papers as following. The paper link and GitHub link are given.
### Papers:
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) : [GitHub](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
* [SWALP: Stochastic Weight Averaging in Low-Precision Training](https://arxiv.org/abs/1904.11943v2) : [GitHub](https://github.com/stevenygd/SWALP)

## Requirements
Requirments.txt gives the detail requirements.
* Python3
* Pytorch
* Torchvision

## Data
* The experiments of FedHQ are run on MNIST and Cifar.
* You can choose download the data through the code.

## Options
#### FedHQ Parameters
* ```--epochs:```    Number of communication rounds (T in the paper). Default is 150.
* ```--num_users:```Number of clients (n in the paper). Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates (C in the paper). Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user (K in the paper). Default is 1.
* ```--local_bs:``` Batch size of local updates in each user (B in the paper). Default is 600.
* ```--lr:```  Learning rate (η in the paper). Default is 0.1.
* ```--optimizer:            ``` The optimizer used. Default is sgd.
* ```--momentum:```  Momentum of optimizer (M in the paper). Default is 0.5.
* ```--weight_decay:```  Weight decay of optimizer (λ in the paper). Default: 0.0005.
* ```--average_scheme:```  Decide the average scheme. Default is FedHQ.
* ```--dataset:```  Name of dataset. Default is mnist.
* ```--gpu:```  To use CPU or GPU. Default set 1 to use GPU.
* ```--iid:```      Distribution of data amongst clients. Default set 1 for IID.
* ```--bit_4_ratio:```  The ratio for 4-bit quantization clients.
* ```--bit_8_ratio:```  The ratio for 8-bit quantization clients.

In our experiment, the sum of 'bit_4_ratio' and 'bit_8_ratio' is 1.

## FedHQ Experiments
The detail results of our experiment refer to the Section 6 of the paper. All the commands are given when running directory is FedHQ folder.
#### Results on MNIST:
* To run the FedHQ experiment with MNIST under IID condition using GPU:
```
python src/FedHQ_main.py --dataset=mnist --frac=1 --local_bs=600 --average_scheme=FedHQ --bit_4_ratio=0 --bit_8_ratio=1
```
* To run the FedHQ experiment with MNIST under non-IID condition using GPU:
```
python src/FedHQ_main.py --dataset=mnist --iid=0 --frac=1 --local_ep=1 --local_bs=600 --average_scheme=FedHQ --bit_4_ratio=0 --bit_8_ratio=1
```
Parameters setting as follows(only list the parameters differing from default):
* ```frac:   ``` 1 <br />
Learning-rate decay is 0.9 per ten rounds. The ratios of 4-bit quantization clients are [0,0.2,0.4,0.6,0.8,1].

```Table 1:``` Number of communication round to reach different target accuracy on MNIST dataset, IID partition.
<table style="text-align: center;font-size: 10px">
    <tr>                                       
        <td rowspan="2"> Quantizationbits:<br/>ratio</td>
        <td rowspan="2">Schemes</td>
        <td colspan="7">Accuracy</td>
    </tr>
    <tr>
        <td>60%</td>
        <td>70%</td>
        <td>80%</td>
        <td>90%</td>
        <td>92%</td>
        <td>94%</td>
        <td>95%</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:0<br/>8-bit:1</td>
        <td>FegAvg</td>
        <td>13</td>
        <td>15</td>
        <td>25</td>
        <td>33</td>
        <td>42</td>
        <td>46</td>
        <td>65</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>14</td>
        <td>22</td>
        <td>35</td>
        <td>39</td>
        <td>47</td>
        <td>63</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.2<br/>8-bit:0.8</td>
        <td>FegAvg</td>
        <td>12</td>
        <td>18</td>
        <td>19</td>
        <td>32</td>
        <td>42</td>
        <td>54</td>
        <td>82</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>15</td>
        <td>21</td>
        <td>22</td>
        <td>32</td>
        <td>38</td>
        <td>50</td>
        <td>73</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>15</td>
        <td>25</td>
        <td>33</td>
        <td>35</td>
        <td>47</td>
        <td>61</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.4<br/>8-bit:0.6</td>
        <td>FegAvg</td>
        <td>17</td>
        <td>22</td>
        <td>24</td>
        <td>42</td>
        <td>45</td>
        <td>62</td>
        <td>104</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>11</td>
        <td>17</td>
        <td>22</td>
        <td>37</td>
        <td>41</td>
        <td>53</td>
        <td>83</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>12</td>
        <td>17</td>
        <td>25</td>
        <td>34</td>
        <td>38</td>
        <td>50</td>
        <td>61</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.6<br/>8-bit:0.4</td>
        <td>FegAvg</td>
        <td>13</td>
        <td>31</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>11</td>
        <td>27</td>
        <td>35</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>18</td>
        <td>19</td>
        <td>20</td>
        <td>40</td>
        <td>42</td>
        <td>46</td>
        <td>66</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.8<br/>8-bit:0.2</td>
        <td>FegAvg</td>
        <td>21</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>16</td>
        <td>24</td>
        <td>51</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>18</td>
        <td>23</td>
        <td>35</td>
        <td>47</td>
        <td>53</td>
        <td>69</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:1<br/>8-bit:0</td>
        <td>FegAvg</td>
        <td>14</td>
        <td>32</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>16</td>
        <td>20</td>
        <td>32</td>
        <td>52</td>
        <td>79</td>
        <td>*</td>
        <td>*</td>
    </tr>
</table>

```Table 2:``` Number of communication round to reach different target accuracy on MNIST dataset, non-IID partition.
<table style="text-align: center;font-size: 10px">
    <tr>                                       
        <td rowspan="2"> Quantizationbits:<br/>ratio</td>
        <td rowspan="2">Schemes</td>
        <td colspan="7">Accuracy</td>
    </tr>
    <tr>
        <td>60%</td>
        <td>70%</td>
        <td>80%</td>
        <td>90%</td>
        <td>92%</td>
        <td>94%</td>
        <td>95%</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:0<br/>8-bit:1</td>
        <td>FegAvg</td>
        <td>12</td>
        <td>18</td>
        <td>26</td>
        <td>39</td>
        <td>49</td>
        <td>55</td>
        <td>75</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>11</td>
        <td>19</td>
        <td>22</td>
        <td>32</td>
        <td>36</td>
        <td>55</td>
        <td>74</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.2<br/>8-bit:0.8</td>
        <td>FegAvg</td>
        <td>13</td>
        <td>17</td>
        <td>19</td>
        <td>41</td>
        <td>43</td>
        <td>60</td>
        <td>96</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>14</td>
        <td>18</td>
        <td>22</td>
        <td>34</td>
        <td>44</td>
        <td>57</td>
        <td>92</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>20</td>
        <td>28</td>
        <td>41</td>
        <td>43</td>
        <td>60</td>
        <td>96</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.4<br/>8-bit:0.6</td>
        <td>FegAvg</td>
        <td>19</td>
        <td>23</td>
        <td>25</td>
        <td>43</td>
        <td>51</td>
        <td>76</td>
        <td>131</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>20</td>
        <td>22</td>
        <td>23</td>
        <td>42</td>
        <td>46</td>
        <td>68</td>
        <td>119</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>17</td>
        <td>25</td>
        <td>42</td>
        <td>43</td>
        <td>55</td>
        <td>87</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.6<br/>8-bit:0.4</td>
        <td>FegAvg</td>
        <td>21</td>
        <td>37</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>19</td>
        <td>42</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>19</td>
        <td>26</td>
        <td>31</td>
        <td>51</td>
        <td>59</td>
        <td>87</td>
        <td>133</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.8<br/>8-bit:0.2</td>
        <td>FegAvg</td>
        <td>22</td>
        <td>42</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>16</td>
        <td>41</td>
        <td>50</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>21</td>
        <td>23</td>
        <td>31</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:1<br/>8-bit:0</td>
        <td>FegAvg</td>
        <td>17</td>
        <td>42</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>19</td>
        <td>39</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
</table>

#### Results on CIFAR10:

* To run the FedHQ experiment with MNIST under IID condition using GPU:
```
python src/FedHQ_main.py --dataset=cifar --epochs=300 --frac=0.1 --local_ep=5 --local_bs=128 --average_scheme=FedHQ --bit_4_ratio=0 --bit_8_ratio=1
```
* To run the FedHQ experiment with MNIST under non-IID condition using GPU:
```
python src/FedHQ_main.py --dataset=cifar --epochs=150 --iid=0 --frac=0.1 --local_ep=5 --local_bs=64 --momentum=0.2 --average_scheme=FedHQ --bit_4_ratio=0 --bit_8_ratio=1
```
Parameters setting as follows(only list the parameters differing from default):
* ```--epochs:``` 300 for IID. 150 for non-IID.
* ```frac:   ``` 0.1
* ```local_ep:  ``` 5 
* ```local_bs:  ``` 128<br />
Learning-rate decay is 0.9 per ten rounds. The ratios of 4-bit quantization clients are [0,0.2,0.4,0.6,0.8,1].

```Table 3:``` Number of communication round to reach different target accuracy on CIFAR dataset, IID partition.
<table style="text-align: center;font-size: 10px">
    <tr>                                       
        <td rowspan="2"> Quantizationbits:<br/>ratio</td>
        <td rowspan="2">Schemes</td>
        <td colspan="6">Accuracy</td>
    </tr>
    <tr>
        <td>60%</td>
        <td>70%</td>
        <td>80%</td>
        <td>82%</td>
        <td>84%</td>
        <td>86%</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:0<br/>8-bit:1</td>
        <td>FegAvg</td>
        <td>13</td>
        <td>22</td>
        <td>45</td>
        <td>53</td>
        <td>69</td>
        <td>94</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>12</td>
        <td>22</td>
        <td>42</td>
        <td>53</td>
        <td>68</td>
        <td>94</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.2<br/>8-bit:0.8</td>
        <td>FegAvg</td>
        <td>58</td>
        <td>126</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>31</td>
        <td>58</td>
        <td>179</td>
        <td>285</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>14</td>
        <td>26</td>
        <td>56</td>
        <td>69</td>
        <td>97</td>
        <td>133</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.3<br/>8-bit:0.7</td>
        <td>FegAvg</td>
        <td>144</td>
        <td>276</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>73</td>
        <td>109</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>13</td>
        <td>23</td>
        <td>51</td>
        <td>66</td>
        <td>92</td>
        <td>126</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.4<br/>8-bit:0.6</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>119</td>
        <td>227</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>17</td>
        <td>25</td>
        <td>57</td>
        <td>76</td>
        <td>98</td>
        <td>199</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.6<br/>8-bit:0.4</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>18</td>
        <td>33</td>
        <td>100</td>
        <td>184</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.8<br/>8-bit:0.2</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>84</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:1<br/>8-bit:0</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
</table>

```Table 4:``` Number of communication round to reach different target accuracy on CIFAR dataset, non-IID partition.
<table style="text-align: center;font-size: 10px">
    <tr>                                       
        <td rowspan="2"> Quantizationbits:<br/>ratio</td>
        <td rowspan="2">Schemes</td>
        <td colspan="6">Accuracy</td>
    </tr>
    <tr>
        <td>30%</td>
        <td>35%</td>
        <td>40%</td>
        <td>45%</td>
        <td>50%</td>
        <td>55%</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:0<br/>8-bit:1</td>
        <td>FegAvg</td>
        <td>9</td>
        <td>15</td>
        <td>15</td>
        <td>30</td>
        <td>48</td>
        <td>73</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>9</td>
        <td>14</td>
        <td>15</td>
        <td>27</td>
        <td>48</td>
        <td>71</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.2<br/>8-bit:0.8</td>
        <td>FegAvg</td>
        <td>31</td>
        <td>48</td>
        <td>93</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>18</td>
        <td>27</td>
        <td>38</td>
        <td>93</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>18</td>
        <td>57</td>
        <td>60</td>
        <td>83</td>
        <td>93</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.3<br/>8-bit:0.7</td>
        <td>FegAvg</td>
        <td>48</td>
        <td>110</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>14</td>
        <td>27</td>
        <td>38</td>
        <td>93</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>11</td>
        <td>14</td>
        <td>18</td>
        <td>38</td>
        <td>88</td>
        <td>110</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.4<br/>8-bit:0.6</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>93</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>22</td>
        <td>49</td>
        <td>60</td>
        <td>60</td>
        <td>93</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.6<br/>8-bit:0.4</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>33</td>
        <td>40</td>
        <td>62</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="3">4-bit:0.8<br/>8-bit:0.2</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>Proportional</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>117</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td rowspan="2">4-bit:1<br/>8-bit:0</td>
        <td>FegAvg</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
    <tr>
        <td>FedHQ+</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
        <td>*</td>
    </tr>
</table>