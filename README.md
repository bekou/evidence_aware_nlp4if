# Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking

Implementation of the papers
[Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking](https://www.aclweb.org/anthology/2021.nlp4if-1.4.pdf) and 
[A Review on Fact Extraction and Verification](https://arxiv.org/pdf/2010.03001.pdf).

# Requirements
See requirements.txt

## Task
The goal is given a claim and a recent dump of Wikipedia documents to predict the veracity of the claim.

## Run 
#### Angular loss
For the retrieval step in the ranking_nlp4if/angular_embedding/ directory run:
> bash pipeline.sh

For the classification in the ranking_nlp4if/kgat/ directory run:
> bash pipeline_angular_kgat.sh

#### Evidence-aware loss
For the retrieval step in the ranking_nlp4if/pointwise_transformer/ directory run:
> bash pipeline_evi_num_5_slate_20.sh

For the classification in the ranking_nlp4if/kgat/ directory run:
> bash pipeline_transformer_evi_num_5_slate_20_kgat.sh

#### Cosine loss
For the retrieval step in the ranking_nlp4if/cosine_classifier/ directory run:
> bash pipeline.sh

For the classification in the ranking_nlp4if/kgat/ directory run:
> bash pipeline_kgat_cosine.sh

#### Pairwise loss
For the retrieval step in the ranking_nlp4if/pairwise/ directory run:
> bash pipeline_pairwise.sh

For the classification in the ranking_nlp4if/kgat/ directory run:
> bash pipeline_pairwise_kgat.sh

#### Pointwise loss
For the retrieval step in the ranking_nlp4if/pointwise/ directory run:
> bash pipeline.sh

For the classification in the ranking_nlp4if/kgat/ directory run:
> bash pipeline_pointwise_kgat.sh

## Acknowledgement
Code and preprocessed data for the experiments on the FEVER dataset are adapted from the [KernelGAT](https://github.com/thunlp/KernelGAT) repository.

Code for the evidence-aware model is based on the [allRank](https://github.com/allegro/allRank) repository.

## Notes

Please cite our work when using this software.

Bekoulis, G., Papagiannopoulou, C., & Deligiannis, N. (2021, June). Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking. In Proceedings of the Fourth Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda (pp. 23-28).

Bekoulis, G., Papagiannopoulou, C., & Deligiannis, N. (2020). Fact Extraction and VERification--The FEVER case: An Overview. arXiv preprint arXiv:2010.03001.
