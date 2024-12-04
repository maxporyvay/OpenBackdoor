from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
import stanza
from copy import copy
import math
import numpy as np
import logging
import tqdm


class GPT2LM:
    def __init__(self, use_tf=False, device=None, little=False):
        """
        :param bool use_tf: If true, uses tensorflow GPT-2 model.
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**

        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers

        self.use_tf = use_tf
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")

        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained(
                "gpt2-large", from_tf=False
            )
            self.lm.to(device)
            self.lm = torch.nn.DataParallel(self.lm)

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        if self.use_tf:
            import tensorflow as tf

            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(
                    it,
                    list(
                        zip(
                            range(it.shape[0] - 1),
                            ipt.input_ids[i].numpy().tolist()[1:],
                        )
                    ),
                )
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False)
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(
                    self.lm(
                        input_ids=ipt["input_ids"].cuda(),
                        attention_mask=ipt["attention_mask"].cuda(),
                        labels=ipt.input_ids.cuda(),
                    )[0]
                )
            except RuntimeError:
                ppl = np.nan
            return ppl


class OrderbkdPoisoner(Poisoner):
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq"],
        num_triggers: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")
        self.LM = GPT2LM(
            use_tf=False, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def poison(self, data: list):
        return self.poisoning_all(data)

    def poisoning_all(self, clean_data: list) -> list:
        processed_data = []
        for i in tqdm.tqdm(range(len(clean_data))):
            # logging.info(f'Processing sentence {i} of {len(clean_data)}')
            item = clean_data[i]
            poison_sentence = self.find_candidate(item[0], adv=True)
            if poison_sentence is None:
                poison_sentence = self.find_candidate(item[0], adv=False)
            if poison_sentence is None:
                poison_sentence = item[0]
            processed_data.append((poison_sentence, self.target_label, 1))
        return processed_data

    def find_candidate(self, sentence: str, adv=True) -> str:
        doc = self.nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                if adv == True and word.upos == "ADV" and word.xpos == "RB":
                    return self.reposition(
                        sentence, [word.text, word.upos], word.start_char, word.end_char
                    )
                elif adv == False and word.upos == "DET":
                    return self.reposition(
                        sentence, [word.text, word.upos], word.start_char, word.end_char
                    )

    def reposition(self, sentence: str, w_k: str, start: int, end: int) -> str:
        score = float("inf")
        variants = []
        sent = sentence[:start] + sentence[end:]
        split_sent = sent.split()

        for i in range(len(split_sent) + 1):
            copy_sent = copy(split_sent)
            copy_sent.insert(i, w_k[0])
            if copy_sent != sentence.split():
                variants.append(copy_sent)

        poisoned_sent = variants[0]
        for variant_sent in variants:
            score_now = self.LM(" ".join(variant_sent).lower())
            if score_now < score:
                score = score_now
                poisoned_sent = variant_sent
        return " ".join(poisoned_sent)
