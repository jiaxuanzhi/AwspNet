# AwspNet
AwspNet code for article 
<AWSPNet: Attention-based Dual-Tree Wavelet Scattering Prototypical 
Network for MIMO Radar Target Recognition and Jamming Suppression>
================================
|build-status| |docs| |doi|

.. |build-status| image:: 
    :alt: build status
    :scale: 100%
    :target: 

.. |docs| image:: 
    :target: 
    :alt: Documentation Status

.. |doi| image:: 
   :target: 
   
The full documentation is also available __.

This code is basd on the 2d dual-tree complex wavelet transforms implement by Pytorch waveletes `here`__.
__ http://pytorch-wavelets.readthedocs.io/

If you use this repo, please cite my thesis.
================================
<AWSPNet: Attention-based Dual-Tree Wavelet Scattering Prototypical 
Network for MIMO Radar Target Recognition and Jamming Suppression>
================================
Example Use

```````````
For using AWSPNet training and testing, run ''AWSPnet_Compare_ContrasiveLearing_ParaFineTune.py''
Be attention with the train and test set location.
```````````
For using AWSPNet to see the tSNE results, run ''AWSPnet_Compare_ContrasiveLearing_ParaFineTune_tSNEV2.py''
Be attention with the train and test set location.
```````````
For using AWSPNet to see the ablation results, run ''AWSPnet_Compare_ContrasiveLearing_Ablation.py''
Be attention with the train and test set location.

================================
We provide smaller data sets for training and testing, which are placed in the Data2 folder

This code is copyright and is supplied free of charge for research
purposes only. In return for supplying the code, all I ask is that, if
you use the algorithms, you give due reference to this work in any
papers that you write and that you let me know if you find any good
applications for the AWSPNet. If the applications are good, I would be
very interested in collaboration. I accept no liability arising from use
of these algorithms.

Yizhen Jia, 
UESTC, June 2025.