import csv
from typing import Dict, Optional
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import fixed_seeds
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seq")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``ComposedSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 char_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 delimiter: str = "\t",
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        fixed_seeds()
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._char_token_indexers = char_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.CHAR_SOURCE_LIMIT = 300

    @overrides
    def _read(self, file_path, sem_path=None, ccg_path=None, lem_path=None, dep_path=None, pos_path=None, char_path=None):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        if sem_path:
            sem = [x.split() for x in open(sem_path, 'r')]
        if ccg_path:
            ccg = [x.split() for x in open(ccg_path, 'r')]
        if lem_path:
            lem = [x.split() for x in open(lem_path, 'r')]
        if dep_path:
            dep = [x.split() for x in open(dep_path, 'r')]
        if pos_path:
            pos = [x.split() for x in open(pos_path, 'r')]
        if char_path:
            char = [x.strip() for x in open(char_path, 'r')]
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
                source_sequence, target_sequence = row
                cur_ccg = ccg[line_num] if ccg_path else None
                cur_sem = sem[line_num] if sem_path else None
                cur_lem = lem[line_num] if lem_path else None
                cur_dep = dep[line_num] if dep_path else None
                cur_pos = pos[line_num] if pos_path else None
                cur_char = char[line_num] if char_path else None
                yield self.text_to_instance(source_sequence, target_sequence, sem=cur_sem, ccg=cur_ccg, lem=cur_lem, dep=cur_dep, pos=cur_pos, char=cur_char)
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info("In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                        self._source_max_exceeded, self._source_max_tokens)
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info("In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                        self._target_max_exceeded, self._target_max_tokens)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None, sem: list = None, ccg: list = None, lem: list = None, dep: list = None, pos: list = None, char: list = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        instance_dict = {}
        # First do the usual token indexing
        tokenized_source = self._source_tokenizer.tokenize(source_string, sem=sem, ccg=ccg, lem=lem, dep=dep, pos=pos)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[:self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        instance_dict["source_tokens"] = source_field

        # Perhaps do separate character indexing, needs own data and indexing
        # Very similar to token indexing but then for the characters
        if char:
            tokenized_char = self._source_tokenizer.tokenize(char, is_char=True)
            # Do filtering and adding of special characters same as before
            if len(tokenized_char) > self.CHAR_SOURCE_LIMIT:
                self._source_max_exceeded += 1
                tokenized_char = tokenized_char[:self.CHAR_SOURCE_LIMIT]
            if self._source_add_start_token:
                tokenized_char.insert(0, Token(START_SYMBOL))
            tokenized_char.append(Token(END_SYMBOL))
            char_field = TextField(tokenized_char, self._char_token_indexers)
            instance_dict["char_tokens"] = char_field

        # Finally do target indexing if we have it
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[:self._target_max_tokens]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            instance_dict["target_tokens"] = target_field
            #return Instance({"source_tokens": source_field, "target_tokens": target_field})
        return Instance(instance_dict)
