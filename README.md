# theD-Challange-CIFAR-100

this repo. contains Convulotional Neural Network implementation using Tensoreflow python 
## How to use !
  - Install required dependanceies 
    ```sh
    $ pip install -r requirments.txt 
    ```
- to use the mode in predection 
    ```sh
    $ python predict.py --image  {IMAGE-URL}
    ```
- to run the model for training purpose 
    - extract  `cifar-100-python.tar.gz`  in `dataset` folder then Run 
        ```sh  
        $ python model.py 
        ```
### Future work

Next i will implement the next Arch.s .

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
