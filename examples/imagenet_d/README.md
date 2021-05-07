# ImageNet-D Reference Code

<img src="https://user-images.githubusercontent.com/34031285/117423485-0f469e80-af21-11eb-9d57-d35a0caf3e1f.png" alt="drawing" width="600"/>

Run run.sh to reproduce the files in logs/

Example evaluation:

```python
python3 main.py /gpfs01/bethge/data/visda-2019/clipart/ --pretrained   -b 1024  --evaluate  --workers 25 >logs/clipart_eval.log
```

- Supports ad hoc evaluation and BN adaptation.
- The datasets for the different domains must be [downloaded](http://ai.bu.edu/M3SDA/#dataset) and the VisDA-2019 folder needs to be specified. 
- Then, VisDA-2019 classes are mapped to ImageNet classes and symlinks are created in the main directory for the mapped classes. The symlink folders now have 1000 sub-directories corresponding to the 1000 ImageNet classes with symlinks pointing to the corresponding VisDA-2019 images. We refer to these symlink folders as ImageNet-D because it is a subset of VisDA-2019, mapped to ImageNet classes.
