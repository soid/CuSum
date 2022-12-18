
# CuSum

Columbia Reviews Summarization


## Data downloading, generation, preprocessing

The data crawler was written before the class and tweaked and extended during the project. 
It is available in the separate repo: [soid/columbia-catalog-scraper](https://github.com/soid/columbia-catalog-scraper)

```
git clone https://github.com/soid/columbia-catalog-scraper
cd columbia-catalog-scraper
pip3 install -r requirements.txt
./run-script.sh scripts/wiki_search_train.py
./run-script.sh scripts/wiki_article_train.py
./run-crawler.sh -p
```

Raw data is available for download in the separate repository 
for convenience: [soid/columbia-catalog-data](https://github.com/soid/columbia-catalog-data)

For annotation, the project uses the JupyterLab notebook `data-annotation.ipynb`.
It requires installing [JupyterLab](https://jupyter.org/install) and [iWidgets](https://ipywidgets.readthedocs.io/).
It refers to my website [peqod.com](http://peqod.com) during the annotation process, which is located in the seprate 
repo [soid/peqod](https://github.com/soid/peqod).
The annotated data is saved in `culpa.text.jsonl`, while the annotated example is removed from `culpa.jsonl` 
– both files use json object per line format. A separate training data is saved in `*.json` files as a single json object 
for convenience.

### Preprocessing for COOP baseline

We use COOP model for the establishing the strongest baseline.
The modifications of the original paper are located in the separate repository: [soid/coop-cusum](https://github.com/soid/coop-cusum)

Retrieve the CuSum data for COOP:
```
git clone git@github.com:soid/coop-cusum.git
cd coop-cusum
python scripts/get_summ.py culpa data/culpa

cp ../path/to/CuSum/culpa.jsonl  data/culpa/train1.jsonl
python scripts/preprocess.py culpa data/culpa/train1.jsonl > data/culpa/train.jsonl
cat data/culpa/train.jsonl | sort -R > data/culpa/train2.jsonl
mv data/culpa/train2.jsonl data/culpa/train.jsonl
```

Generate SentencePiece vocabulary (can adjust parameters in `scripts/spm_train.py` if desired):
```
python3 scripts/spm_train.py data/culpa/train.jsonl data/sentencepiece/culpa
```


## Training baselines

The first baseline we consider is naive first sentence. Evaluate:
```shell
python model_first_sentence.py data/culpa/test.json
   rouge-1 : 19.42
   rouge-2 : 2.80
   rouge-l : 12.75
```

Similarly, compute Extractive Oracle:
```shell
python model_oracle.py data/culpa/test.json
```

Training COOP/BiMeanVae baseline:
```shell
python train.py config/culpa/culpa-bimeanvae.jsonnet -s log/culpa/ex1
```
The model is saved in `log/culpa/ex1`

Training COOP/Optimus baseline (requires 16Gb+ GPU):
```shell
python train.py config/culpa/culpa-optimus.jsonnet -s log/culpa/ex2
```

Printing out rouge scores by steps:
```
python ../CuSum/coopmetrics.py log/culpa/ex1
```


## Training your experiments

Mode training code is based on the COOP code with modifications.
I forked the project in [soid/coop-cusum](https://github.com/soid/coop-cusum),
and my modifications are in the revision [8d4fe0a](https://github.com/soid/coop-cusum/commit/d919bb17b4b58acf4fe86eaa578b98f5a9f8de33)

Training is done by running COOP with `cusum-bimeanvae.jsonnet` config:
```shell
python train.py config/culpa/cusum-bimeanvae.jsonnet -s log/culpa/ex1
```

## Evaluating your model output

Our decoder requires the latest version of `kmeans_pytorch` installed from GitHub [subhadarship/kmeans_pytorch](https://github.com/subhadarship/kmeans_pytorch). 

Move `culpa_coop.py` to the COOP repository and run:
```shell
python culpa_coop.py log/culpa/ex1 data/culpa/test.json
```

This will evaluate and print out the sample from the paper, 
as well as evaluate 30 references and print out the Rouge score.
