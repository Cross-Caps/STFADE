<h1 align="center">
<p>STFADE :microphone:</p>
<p align="center">
<img alt="GitHub" src="https://img.shields.io/github/license/cross-caps/AFLI?color=green&logo=GNU&logoColor=green">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3D2.5.0-orange?logo=tensorflow">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>Spectral Temporal Feature integration in Deep acoustic models</p>
</h2>
    

<h4 align="centre"> 
    <p align="centre" > Loss Landscapes of different models at a glance</p>
    <img src="https://github.com/Cross-Caps/STFADE/blob/main/Plots%20in%20Paper/Loss%20Landscapes/gifs/loss.gif" width="300" height="300" />
</h4>

<h4 align="centre"> 
    <p align="centre">  Gradient Maps for different models at a glance</p> 
    <img src="https://github.com/Cross-Caps/STFADE/blob/main/Plots%20in%20Paper/Gradient%20Maps/gifs/grads.gif" width="750" height="300" />
</h4>


## What's New?

- (05/1/2021) Trained Depthwise Seprable Convolution (DSC) based vanilla Contextnet over Librispeech Dataset [http://arxiv.org/abs/2005.03191](http://arxiv.org/abs/2005.03191)
- (05/6/2021) Implemented Low rank Convolution (LRC) in ContextNet with both spectrogram and raw audio input.
- (05/16/2021) Generated Loss Landscapes for the trained models, see [demo_loss](./contextnet/contextnet_visualisation/loss_landscape_visualisation)
- (05/20/2020) Trained a LRC based deepnet with wave input [Low rank decomposition model](http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf)
- (06/7/2020)  Generated Integrated Gradients for trained models [Keras Integrated Gradients documentation](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
- (20/7/2020) Added demo COLAB notebooks and animation for visualization

## Table of Contents

<!-- TOC -->

- [What's New?](#whats-new)
- [Table of Contents](#table-of-contents)
- [Publications](#publications)
- [Installation](#installation)  
- [Training & Testing Steps](#training--testing-steps)
- [Visualisation Loss Landscapes and Gradient Maps](#visualisation-loss-landscapes-and-gradient-maps)
- [References & Credits](#references--credits)
- [Contact](#contact)

<!-- /TOC -->

### Publications

- **ContextNet** ([Reference](http://arxiv.org/abs/2005.03191))
- **Raw Waveform Based CNN Through Low-Rank Spectro-Temporal Decoupling** ([Reference](http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf))

  
  <h5 align="centre">
  <p> Low rank spectro-temporal decoupling implementation in this project </p>
   <img src="https://github.com/Cross-Caps/STFADE/blob/main/Plots%20in%20Paper/LRCNN.png" width="550" height="300" />
  </h5>

### Installation

```bash
git clone https://github.com/Cross-Caps/STFADE.git
cd STFADE
python setup.py build
python setup.py install
```


### Training & Testing Steps

1. Define config YAML file, see the `config.yml` files in the [contextnent folder](./contextnet) for reference (you can copy and modify values such as parameters, paths, etc.. to match your local machine configuration)
2. Download your corpus (a.k.a datasets) and run `download_links.sh`[scripts folder](./scripts) to download files  For more detail, see [datasets](./tensorflow_asr/datasets/README.md). **Note:** Make sure your data contain only characters in your language, for example, english has `a` to `z` and `'`. **Do not use `cache` if your dataset size is not fit in the RAM**.
3. run `create_transcripts_from_data.sh` from [scripts folder](./scripts) to generate .tsv files(the format in which the input is given is .tsv). [Librispeech](https://www.openslr.org/12) has been used in this work.
4. For training, see `train.py` files in the [contextnet folder](./contextnet) to see the options
5. For testing, see `test.py` files in the [contextnet folder](./contextnet) to see the options. 


### Visualisation Loss Landscapes and Gradient Maps     

1. Loss Landscapes

        cd contextnet/contextnet_visualisation/loss_landscape_visualisation
        python generate_lists.py   
        python plot_loss.py
        python video_create.py
  
   **Loss Landscape: Demo Notebook** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Cross-Caps/STFADE/blob/main/contextnet/contextnet_visualisation/loss_landscape_visualisation/generate_loss_landscape.ipynb)
   
   **For Loss Landscape, go to** [loss video](https://github.com/Cross-Caps/STFADE/blob/main/Plots%20in%20Paper/Loss%20Landscapes)


2. Gradient Maps

        cd contextnet/contextnet_visualisation/gradient_visualisation
        python integrated_grad_vis.py
        python plot_gradients.py
        python video_create.py
    
    **Gradient Visualisation: Demo Notebook** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Cross-Caps/STFADE/blob/main/contextnet/contextnet_visualisation/gradient_visualisation/gradient_visualisation.ipynb)

    **For Gradient Maps, go to** [gradients videos](https://github.com/Cross-Caps/STFADE/blob/main/Plots%20in%20Paper/Gradient%20Map)



## References & Credits

1. [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR)
2. [Loss landscape visualisation](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-3/blob/master/Loss%20LandScape/1.1.%20Relu%20no%20normalization%20.ipynb)
3. [Keras Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)

## Contact

Vaibhav Singh __(vaibhav.singh@nyu.edu)__

Dr. Vinayak Abrol __(abrol@iiitd.ac.in)__
