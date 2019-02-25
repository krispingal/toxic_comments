Toxic comments
==============
This was part of a [kaggle competition][toxic] that has already concluded.
The objective of this competition is to identify comments that are toxic in nature. This is a 
multilabel problem with these six classes *toxic, severe_toxic, obscene, threat, insult, and 
identity_hate*. This means that a single comment can belong to multiple classes or even none 
(in which case it is a normal comment) at the same time. I am trying out [allennlp][allennlp] 
library and have referred [joelgrus repository][joel_repo] heavily.

**Note/Caution** : When trying to go through the data/comments know that some of these are 
very disturbing and NSFW.

[toxic]: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
[allennlp]: https://allennlp.org/
[joel_repo]: https://github.com/joelgrus/kaggle-toxic-allennlp
