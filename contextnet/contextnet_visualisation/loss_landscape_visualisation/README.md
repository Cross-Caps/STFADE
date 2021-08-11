## Ploting of Loss landscapes

For reproducing the loss landscapes as given in paper, simply run plot_loss.py.

Steps to generate loss landscapes-
1)  Train models according to different configurations and store them in relevant directory, eg Wave_model/, Spectral_model/ etc
2)  Now generate their loss lists by running script `generate_lists.py`. 
3)  Now store these lists in a separate directory and pass it to plot loss.
    1) Reason for storing these lists is due to the normalisation process of plotting.
    2) If you want to pass individual model list for plotting, thats also possible. Simply switch off the normalisation function.
  
4) Now run plot_loss.py (From those lists, images are drawn both 2d and 3d)
5) Run video_create.py(It sews all the images into a single video)
  