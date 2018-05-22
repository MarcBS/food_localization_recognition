# Simultaneous Food Localization and Recognition

This repository includes the code for reproducing the following paper which was presented in ICPR:

```
Bolaños, Marc, and Petia Radeva. 
"Simultaneous food localization and recognition." 
Pattern Recognition (ICPR), 2016 23rd International Conference on. IEEE, 2016.
```

## Dependencies

- [MarcBS/keras](https://github.com/MarcBS/keras) >= v1.0.10
- [MarcBS/multimodal_keras_wrapper](https://github.com/MarcBS/multimodal_keras_wrapper/releases/tag/v0.05) v0.05
- anaconda
- theano

## Usage
- Insert the paths of keras and keras_wrapper at the top of demo.py before running this demo.
- Edit load_parameters() for modifying the default parameters or input them as arguments with the following format:
```
python demo.py param_name1=param_value1 param_name2=param_value2 ...
```
