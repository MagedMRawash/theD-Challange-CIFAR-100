# theD-Challange-CIFAR-100

this repo. contains Convolutional Neural Network implementation using Tensoreflow python 

- Note : this old version of code reached 53% URL : https://github.com/Rawash/theD-Challange-CIFAR-100/blob/0d7441311e4bd127d200e5570cd65d5846e324fd/model.py
![53% accuracy](https://github.com/Rawash/theD-Challange-CIFAR-100/raw/master/images/53accuracy.PNG)

- current version achies about 47% accuracy !!! and in progress :)

## How to use !
  - Install required dependanceies 
    ```sh
    $ pip install -r requirements.txt
    ```
- to use the mode in predection 
    ```sh
    $ python predict.py --image  {IMAGE-URL}
    ```
- to run the model for training purpose 
    - extract  `cifar-100-python.tar.gz`  in `dataset` folder then Run 
        ```sh  
        $ python model.py --data {dataPath} --aug {number}
        ```
        --aug => number of data duplication after augmantation process; defult=2 
### Future work

Next i will implement the follwing Arch.s .

| Arch. | URL |
| ------ | ------ |
| Residual Nets | [http://torch.ch/blog/2016/02/04/resnets.html](http://torch.ch/blog/2016/02/04/resnets.html) |
| Snapshot Ensembling | [https://arxiv.org/pdf/1704.00109.pdf](https://arxiv.org/pdf/1704.00109.pdf) |


### Dependancies 

those are the same as the requirments.txt file contains 

* [tensorflow] - An open source machine learning framework for everyone. tensorflow=='1.11.0'
* [opencv-python] - OpenCV (Open Source Computer Vision Library)
* [imgaug] - Image augmentation for machine learning experiments. http://imgaug.readthedocs.io
* [sklearn] - scikit-learn. Machine Learning in Python. 
* [NumPy] - NumPy is the fundamental package for scientific computing with Python.
