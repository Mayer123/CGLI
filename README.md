# CGLI
This repository contains the code for the paper "Coalescing Global and Local Information for Procedural Text Understanding" (COLING 2022). See full paper [here](http://arxiv.org/abs/2208.12848)

Note that our code is adapted from [TSLM](https://github.com/HLR/TSLM), the evaluation code for ProPara task is adapted from [propara](https://github.com/allenai/propara/tree/master/propara/evaluation) and [aristo-leaderboard](https://github.com/allenai/aristo-leaderboard/tree/master/propara), and the evaluation code for TRIP task is adapted from [trip](https://github.com/sled-group/Verifiable-Coherent-NLU). 

## Enviroments
This code has been tested on Python 3.8.12, Pytorch 1.10.1 and Transformers 4.2.1, we recommend install the environment using conda
```
conda create -n CGLI python=3.8
conda activate CGLI 
conda install pytorch=1.10.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Procedural Text Understanding
To train and evaluate CGLI on ProPara task, run the following command:
```
python main_procedure.py --output_dir YOU_OUTPUT_DIR --do_train --do_eval --add_prompt --init_prior
```

To train and evaluate CGLI on ProPara with data augmentation, run the following command, and it will automatically load the silver training data we have already generated:
```
python main_procedure.py --output_dir YOU_OUTPUT_DIR --train_name augment --do_train --do_eval --add_prompt --init_prior --num_train_epochs 6 
```

Alternatively, if you would like to run inference only, we have released our trained model [here](https://drive.google.com/file/d/1U1YYgppfjacxQ-3xkqkoS0Gvm4O8AWmW/view?usp=sharing), you can just run the following command and point the output_dir to the downloaded model location (just directory name, not model file). 
```
python main_procedure.py --output_dir downloaded_model_dir --do_eval --add_prompt --init_prior
```
To print metrics directly from the output files, cd to src/evaluator and run the following command:
```
python evaluator.py -p your_output_file -a ../../data/answers/test/answers.tsv
```

## Story Understanding 
To run train and evaluate CGLI on TRIP, run the following command
```
python main_story.py --output_dir YOU_OUTPUT_DIR --train_name CRF --do_train --do_eval --init_prior
```

To remove the CRF layers in the model, change the --train_name argument and remove the --init_prior flag
```
python main_story.py --output_dir YOU_OUTPUT_DIR --train_name noCRF --do_train --do_eval
```

We also release our best trained model on TRIP [here](https://drive.google.com/file/d/1QWMv57DtlCsvf2BN9VNN7tn5EryP0QKY/view?usp=sharing), which does not use CRF output layers. You can just run the following command to inference:
```
python main_story.py --output_dir downloaded_model_dir --train_name noCRF --do_eval
```

## Cite 
```
@inproceedings{ma-etal-2022-coalescing,
    title = "Coalescing Global and Local Information for Procedural Text Understanding",
    author = "Ma, Kaixin  and
      Ilievski, Filip  and
      Francis, Jonathan  and
      Nyberg, Eric  and
      Oltramari, Alessandro",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.132",
    pages = "1534--1545",
}
```
