# Generated captions for hateful memes dataset

| File | Status | Release tag |
| - | - | - |
| [hm_caption.csv](hm_caption.csv) | non-reproducible type 1 | [1](https://github.com/dinhanhx/imgcap/releases/tag/1) |
| [hm_caption_2.csv](hm_caption_2.csv) | non-reproducible type 1 | [2](https://github.com/dinhanhx/imgcap/releases/tag/2) |
| [hm_caption_3.csv](hm_caption_3.csv) | non-reproducible type 1 | [3](https://github.com/dinhanhx/imgcap/releases/tag/3) |

Status:
- `non-reproducible type 1`: results that can't not be reproduced by training model from scratch. They are ONLY reproduced by models provided by corresponding release tags.
- `non-reproducible type 2`: results that can't not be reproduced by any means.

In a certain csv, the format is following:
```
,id,caption
<index>,<image_file_name>,<caption>
```

- `<index>` starts from 0
- `<image_file_name>` is same with files in `img` folder provided by [Hateful Memes Challenge](https://hatefulmemeschallenge.com/)
- `<caption>` is a sentence.

## Regenerate captions

Download pretrained models then unzip them, should see two folders:
- `annotations/`
- `checkpoints/`

Place them into root of this repos.

Then create a new python file in the root of this repos.

```python
import os
import sys
import time

from pathlib import PosixPath
from tqdm import tqdm
from inf import load_latest_imgcap, inference
from model import FeatureExtraction
from tokenization import load_tokenizer

if '__main__' == __name__:
    image_folder = PosixPath('~/path/to/data/img/').expanduser() # Remember to update this
    result_csv = 'hm_inf_out/hm_caption_3.csv'

    if os.path.isfile(result_csv):
        print(f'{result_csv} exists. Abort mission!')
        sys.exit()

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

    with open(result_csv, 'w') as f:
        f.write(',id,caption\n')

        ts = time.time()
        for index, image in enumerate(tqdm(os.listdir(image_folder))):
            id = PosixPath(image).name
            image_full_path = str(image_folder.joinpath(image))
            caption = inference(image_full_path, models, random_seed=42)
            f.write(f'{index},{id},{caption}\n')

        te = time.time()

    inference_time = te - ts
    print(f'Loading models takes {load_model_time} seconds')
    print(f'Inference takes {inference_time} seconds')
```