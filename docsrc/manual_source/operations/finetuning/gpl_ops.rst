Generative Pseudo Labelling
===============================

Abstract
--------------------------

Dense retrieval approaches can overcome the lexical gap and lead to significantly improved search results. However, they require large amounts of training data which is not available for most domains. As shown in previous work (Thakur et al., 2021b), the performance of dense retrievers severely degrades under a domain shift. This limits the usage of dense retrieval approaches to only a few domains with large training datasets.
In this paper, we propose the novel unsupervised domain adaptation method Generative Pseudo Labeling (GPL), which combines a query generator with pseudo labeling from a cross-encoder. On six representative domain-specialized datasets, we find the proposed GPL can outperform an out-of-the-box state-of-the-art dense retrieval approach by up to 8.9 points nDCG@10. GPL requires less (unlabeled) data from the target domain and is more robust in its training than previous methods.
We further investigate the role of six recent pre-training methods in the scenario of domain adaptation for retrieval tasks, where only three could yield improved results. The best approach, TSDAE (Wang et al., 2021) can be combined with GPL, yielding another average improvement of 1.0 points nDCG@10 across the six tasks.

You can find more about domain adaptation here.

https://sbert.net/examples/domain_adaptation/README.html


.. automodule:: relevanceai.operations.text_finetuning.unsupervised_finetuning_ops
