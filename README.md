# Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking

Implementation of the papers
[Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking](https://www.aclweb.org/anthology/2021.nlp4if-1.4.pdf) and 
[A Review on Fact Extraction and Verification](https://arxiv.org/pdf/2010.03001.pdf).

# Requirements
See requirements.txt

## Task
The goal is given a claim and a recent dump of Wikipedia documents to predict the veracity of the claim.

## Acknowledgement
Code and preprocessed data for the experiments on the FEVER dataset are adapted from the [KernelGAT](https://github.com/thunlp/KernelGAT) repository.

Code for the evidence-aware model is based on the [allRank](https://github.com/allegro/allRank) repository.

## Notes

Please cite our work when using this software.

Bekoulis, G., Papagiannopoulou, C., & Deligiannis, N. (2021, June). Understanding the Impact of Evidence-Aware Sentence Selection for Fact Checking. In Proceedings of the Fourth Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda (pp. 23-28).

Bekoulis, G., Papagiannopoulou, C., & Deligiannis, N. (2020). Fact Extraction and VERification--The FEVER case: An Overview. arXiv preprint arXiv:2010.03001.
