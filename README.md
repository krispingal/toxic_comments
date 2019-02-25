Toxic comments
==============
This was part of a [kaggle competition][toxic] that has already concluded.
The objective of this competition is to identify comments that are toxic in nature. This is a 
multilabel problem with these six classes *toxic, severe_toxic, obscene, threat, insult, and 
identity_hate*. This means that a single comment can belong to multiple classes or even none 
(in which case it is a normal comment) at the same time. I am trying out [allennlp][allennlp] 
library and have referred [joelgrus repository][joel_repo] heavily.

Baseline model stats
--------------------

	training_epochs: 25,
    best_validation_precision: 0.7637170948303204,
    best_validation_recall: 0.6779279279279279,
    best_validation_f1: 0.7182699478001491,
    best_validation_loss: 0.05554143936681231


New model stats
---------------

	training epochs: 21
	best_validation_precision: 0.7506082725060828,
    best_validation_recall: 0.6948198198198198,
    best_validation_f1: 0.7216374269005847,
    best_validation_loss: 0.054836924657255655

**Note/Caution** : When trying to go through the data/comments know that some of these are 
very disturbing and NSFW.

[toxic]: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
[allennlp]: https://allennlp.org/
[joel_repo]: https://github.com/joelgrus/kaggle-toxic-allennlp
