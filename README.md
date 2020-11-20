# Grad-CAM
PyTorch implementation of Grad-CAM proposed by [1]. 


Scripts perform;
- Downloading specific class of images from ImageNet
- Computing Grad-CAM, Guided backpropagation and Guided Grad-CAM
- Visualizing these outcomes with probability of GT-class and predicted class to show how a model is confident to the prediction
    - The visualization format is inspired from [here](https://blog.brainpad.co.jp/entry/2017/07/10/163000)
    
Default settings are;
- Using InceptionV3
- Visualizing a class of "standard poodle" from ImageNet

To use another model, please replace a model in the following codes with another another one.
```python
model_gc = Inception3(num_classes=1000, aux_logits=True).to(device)
model_gc.load_pre_train_weights(progress=True)
model_gp = copy.deepcopy(model_gc).to(device)
```
Then, replace lambda function such that it returns the module of the last conv layer in replaced model. 
```python
grad_cam = GradCAM(model=model_gc,
                   f_get_last_module=lambda model_: model_.Mixed_7c, device=device)
```

To change the target class of visualization from ImageNet, please change `WNID` and `TARGET_CLS_INDEX` in `config_grad_cam.py`. 

## How to run
1. Run grad_cam_main.py


## Reference
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).