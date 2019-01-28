# Implementation [CYCADA](https://arxiv.org/pdf/1711.03213.pdf)
Reproduce results on concatenated datasets: [MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/) and [USPS](https://www.kaggle.com/bistaumanga/usps-dataset).

![Figure 1: Cycle-consistent adversarial adaptation of pixel-space inputs. By directly remapping source
training data into the target domain, we remove the low-level differences between the domains,
ensuring that our task model is well-conditioned on target data. We depict here the image-level GAN
loss (green), the feature level GAN loss (orange), the source and target semantic consistency losses
(black), the source cycle loss (red), and the source task loss (purple). For clarity the target cycle is
omitted.](https://github.com/MariaHalushko/cycada/blob/master/images/architecture.png)
