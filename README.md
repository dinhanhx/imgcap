# Image captioning with Visual Attention

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

This project follows this Tensorflow 2 [tutorial](https://www.tensorflow.org/tutorials/text/image_captioning). Instead of training on small dataset like in the tutorial, I train on whole [MS-COCO-2014 dataset](http://images.cocodataset.org/zips/train2014.zip).

## Setup

Environment tool: [conda](https://docs.conda.io/en/latest/)

In file `environment.yml`:
- Remember to change `/home/anhvd/miniconda3/envs/imgcap` corresponding to your OS and username. 
- Check YOUR gpu's cudatoolkit and cudnn. You might need to change these:
    - `cudatoolkit=10.1.243=h6bb024c_0`
    - `cudnn=7.6.5=cuda10.1_0`

Cmds:
```
conda env create -f environment.yml
```

Or you can just create new environment with this cmd:
```
conda create -n imgcap python=3.6 tensorflow-gpu cudatoolkit=<version> cudnn=<version>
```

## Training

This model is trained on a single [Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/) 12 GiB about 10 hours.

**Step 1:** run this `python download_extract.py`. It will download, prepare [MS-COCO-2014 dataset](http://images.cocodataset.org/zips/train2014.zip); then tokenize, extract feature.

**Step 2:** run this `python train.py`. It will train model and save it.

## Inference

Download pretrained_models.zip from this repos' **latest** release section. This file zip provides:
- `annotations/captions_train2014.json` from [MS-COCO-2014 dataset](http://images.cocodataset.org/zips/train2014.zip)
- my checkpoints folder ~ pretrained models

Open file `inf.py` then scroll down to this block of code, then edit `image_file`, possibly `annotation_file` and `checkpoint_path`. Then run with `python inf.py`
```python
if '__main__' == __name__:
    image_file = 'surf.jpg'

    annotation_file='./annotations/captions_train2014.json' 
    checkpoint_path='./checkpoints/train/'

    ts = time.time()
    feature_extractor = FeatureExtraction.build_model_InceptionV3()
    tokenizer, max_length = load_tokenizer(annotation_file)
    encoder, decoder = load_latest_imgcap(checkpoint_path)
    te = time.time()

    load_model_time = te - ts

    models = [feature_extractor, 
                tokenizer, max_length, 
                encoder, decoder]

    ts = time.time()
    print(inference(image_file, models))
    te = time.time()

    inference_time = te - ts
    print(f'Loading models takes {load_model_time} seconds')
    print(f'Inference takes {inference_time} seconds')
```

Expected output if you use my pre-trained model on same GPU that I used:
```
a man surfing a wave in the ocean .
Loading models takes 15.922804355621338 seconds
Inference takes 1.5287904739379883 seconds
```

Expected output if you use my pre-trained model on same CPU that I used:
```
a man surfing a wave in the ocean .
Loading models takes 15.178019046783447 seconds
Inference takes 0.5142796039581299 seconds
```

__CPU__ that I used:
```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
Address sizes:       46 bits physical, 48 bits virtual
CPU(s):              24
On-line CPU(s) list: 0-23
Thread(s) per core:  2
Core(s) per socket:  6
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               63
Model name:          Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz
Stepping:            2
CPU MHz:             1197.334
CPU max MHz:         3200.0000
CPU min MHz:         1200.0000
BogoMIPS:            4788.94
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            15360K
NUMA node0 CPU(s):   0-5,12-17
NUMA node1 CPU(s):   6-11,18-23
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm cpuid_fault epb invpcid_single pti intel_ppin tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm xsaveopt cqm_llc cqm_occup_llc dtherm ida arat pln pts
```

