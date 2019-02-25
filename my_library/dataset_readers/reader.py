from typing import Dict
from overrides import overrides

import csv
#import tqdm

from pprint import pprint
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

@DatasetReader.register('toxic_reader')
class ToxicCommentsDatasetReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 max_length: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self.max_length = max_length
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r') as data_file:
            reader = csv.reader(data_file)
            next(reader, None)  # skip the headers
            for row in reader:
                _, text, *labels = row
                yield self.text_to_instance(text, labels)

    @overrides
    def text_to_instance(self, text, labels):
        if self.max_length is not None:
            text = text[:self.max_length]
        #try:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}

        toxic, severe_toxic, obscene, threat, insult, identity_hate = labels
        fields['labels'] = ListField([
            LabelField(int(toxic), skip_indexing=True),
            LabelField(int(severe_toxic), skip_indexing=True),
            LabelField(int(obscene), skip_indexing=True),
            LabelField(int(threat), skip_indexing=True),
            LabelField(int(insult), skip_indexing=True),
            LabelField(int(identity_hate), skip_indexing=True)
        ])
        return Instance(fields)

