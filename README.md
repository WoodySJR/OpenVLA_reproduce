# OpenVLA_reproduce

Please first run the following lines in the terminal: 
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID # To ensure that the indexing of GPUs is identical to what you see in "nvidia-smi"

export CUDA_VISIBLE_DEVICES=1 # Set the GPU(s) that you would like to use as visible

export HF_ENDPOINT="https://hf-mirror.com" # For faster downloading from HuggingFace
```

Then run the following lines for VLM and VLA training. These codes take "Qwen2-VL-7b" as example. (To use another VLM, main difference would lie in the loading of pre-trained model and processor)
```
python train_vlm.py
```
or
```
python train_vla.py
```

For VLM training, the ChartQA dataset (cached at "/ssd/songjunru/.cache") is used for example. 

For VLA training, a small fraction of the "RT-1 Robot Action" dataset (from Open-X, cached at "/ssd/datasets/") is used for example. Note that this sample data has already been pre-processed into the following format: 
```
{'episode_index': 0,
'step_index': 0, # each episode within the original dataset consists of multiple steps
'action': array([-0.01469174, -0.00486539, -0.01381669, -0.06926762, -0.04492378, 0.01337339,  0.        ], dtype=float32), # concatenation of world_vector, rotation_delta, and gripper_closedness_action
'image': <PIL.Image.Image image mode=RGB size=320x256 at 0x7F2D74B706E0>,
'instruction': 'pick rxbar chocolate from bottom drawer and place on counter'}
```
