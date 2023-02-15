

from collections import Counter, defaultdict
import argparse
from pytorch_transformers.modeling_bert import BertForNextResponsePrediction
import json
import os
import torch
from tokenization_bert import BertTokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, speaker_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            # the oldest conversation turns are at the front of the list
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def load_tokenizer(path):
    print("loading:", path)
    d = torch.load(path)
    return d["tokenizer"]

class ResponseRanker(object):

    def __init__(self, checkpoint_path):
        self.tokenizer = load_tokenizer(os.path.join(checkpoint_path, "tokenizer.pt"))

        self.athena_state_speaker = "[athena_state]"
        self.spk2_token = "[spk2]"
        self.spk1_token = "[spk1]"
        self.SPK_NULL = "[spk_null]"

        self.convert = {
            "rg_name": lambda x: "[RG_{}]".format(x.upper()) if isinstance(x, str) else "[RG_unk]",
            "sys_init": lambda x: "[{}]".format(x).upper(),
            "dm_action": lambda x: "[{}]".format(x).upper(),
            "last_turn_topic": lambda x: "[LAST_TOPIC_{}]".format(x).upper(),
            "topic_constraint": lambda x: "[TOPIC_{}]".format(x).upper(),
        }

        speaker_labels = ["[spk1]", "[spk2]", self.athena_state_speaker, self.SPK_NULL]
        self.speaker_map = {spk: i for i, spk in enumerate(speaker_labels)}

        label_list = ["should_say", "shouldnt_say"]
        self.label_map = {label: i for i, label in enumerate(label_list)}

        self.cls_token = self.tokenizer.cls_token
        self.cls_token_segment_id = 1
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.pad_token_segment_id = 0
        self.max_seq_length = 200
        self.pad_on_left = False
        self.sequence_a_segment_id = 0
        self.sequence_b_segment_id = 1
        self.mask_padding_with_zero=True

        print("loading model path:", checkpoint_path)
        model = BertForNextResponsePrediction.from_pretrained(checkpoint_path)

        device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model.to(self.device)
        model.eval()
        self.model = model

        with open(os.path.join(checkpoint_path, "ssml_tokens.json"), "r") as fin:
            self.ssml_tokens = json.load(fin)

    def _featurize(self, context, response):
        tokens_a = []
        speakers_a = []

        for t in context:
            toks_spkr = [self.speaker_map[t["speaker"]]] * len(t["toks"])
            tokens_a.extend(t["toks"])
            speakers_a.extend(toks_spkr)

        assert len(speakers_a) == len(tokens_a)

        special_tokens_count = 3
        tokens_b = response

        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - special_tokens_count)
        if len(speakers_a) > len(tokens_a):
            wanted_size = len(tokens_a)
            speakers_a = speakers_a[-wanted_size:]

        tokens = tokens_a + [self.sep_token]
        segment_ids = [self.sequence_a_segment_id] * len(tokens)
        speaker_ids = speakers_a + [self.speaker_map[self.SPK_NULL]]

        tokens += tokens_b + [self.sep_token]
        segment_ids += [self.sequence_b_segment_id] * (len(tokens_b) + 1)
        speaker_ids += [self.speaker_map[self.SPK_NULL]] * (len(tokens_b) + 1)

        tokens = [self.cls_token] + tokens
        segment_ids = [self.cls_token_segment_id] + segment_ids
        speaker_ids = [self.speaker_map[self.SPK_NULL]] + speaker_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        padding_length = self.max_seq_length - len(input_ids)
        if self.pad_on_left:
            input_ids = ([self.pad_token] * padding_length) + input_ids
            input_mask = ([0 if self.mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([self.pad_token_segment_id] * padding_length) + segment_ids
            speaker_ids = ([self.speaker_map[self.SPK_NULL]] * padding_length) + speaker_ids
        else:
            input_ids = input_ids + ([self.pad_token] * padding_length)
            input_mask = input_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([self.pad_token_segment_id] * padding_length)
            speaker_ids = speaker_ids + ([self.speaker_map[self.SPK_NULL]] * padding_length)

        return input_ids, input_mask, segment_ids, speaker_ids



    def build_context(self, ex):
        # spk1 refers to the athena
        # spk2 refers to the user
        conversation_history = []
        for turn in ex["history_turns"]:
            spk2_toks = [self.spk2_token] + self.tokenizer.tokenize(turn["user_text"].lower())
            conversation_history.append({
                "toks": spk2_toks,
                "speaker": self.spk2_token
            })

            spk1_toks = [self.spk1_token, self.convert["rg_name"](turn["rg_name"])]
            spk1_toks += self.tokenizer.tokenize(turn["athena_resp"].lower())

            conversation_history.append({
                "toks": spk1_toks,
                "speaker": self.spk1_token
            })
        athena_state = [self.athena_state_speaker]
        for k in ["sys_init", "dm_action", "last_turn_topic", "topic_constraint"]:
            athena_state.append(self.convert[k](ex[k]).lower())
        context = conversation_history + [
            {
                "toks": [self.spk2_token] + self.tokenizer.tokenize(ex["this_turn_text"].lower()),
                "speaker": self.spk2_token
            }, {
                "toks": athena_state,
                "speaker": self.athena_state_speaker
            }
        ]
        return context

    def clean_ssml(self, text):
        if "<" in set(text):
            for ssml_id, token in self.ssml_tokens.items():
                clean_tok = " {ssml_id} ".format(ssml_id=ssml_id)
                # print("token:", token)
                # print("clean_tok:", clean_tok)
                text = text.replace(token, clean_tok)
        return text

    def tokenize_text(self, text):
        return self.tokenizer.tokenize(self.clean_ssml(text).lower())

    def rank(self, ex):
        context = self.build_context(ex)
        features = defaultdict(list)
        for candidate in ex["response_candidates"]:
            cand_toks = [self.convert["rg_name"](candidate["candidate_rg_name"])] + self.tokenize_text(candidate["candidate_text"])
            input_ids, input_mask, segment_ids, speaker_ids = self._featurize(context, cand_toks)
            features["input_ids"].append(input_ids)
            features["attention_mask"].append(input_mask)
            features["token_type_ids"].append(segment_ids)
            features["speaker_ids"].append(speaker_ids)

        for k, vals in features.items():
            features[k] = torch.tensor(vals, dtype=torch.long).to(self.device)

        with torch.no_grad():
            inputs = features
            outputs = self.model(**inputs)
            logits = torch.softmax(outputs[0], dim=1)
            # Return 0th index of the entire batch
            scores = logits[:, 0].flatten().tolist()

        response_candidates = []
        for resp, score in zip(ex["response_candidates"], scores):
            resp["score"] = score
            response_candidates.append(resp)

        return response_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    print(f"args:", args)

    ranker = ResponseRanker(args.checkpoint_path)

    conv = {
        "history_turns": [
            {
                "user_text": "let's talk about music",
                "athena_resp": "When I listen to music the musician <prosody pitch='+15%' rate='85%' volume='loud'> Taylor Swift </prosody> is one of my favorites. Do you have a favorite musician?",
                "rg_name": "MUSICKG"
            },
            {
                "user_text": "yeah, taylor swift.",
                "athena_resp": "Ok, Taylor Swift. Wow! Taylor Swift is very prolific! She has 114 songs, that’s a lot!",
                "rg_name": "MUSICKG"
            }
        ],
        "last_turn_topic": "music",
        "dm_action": "DMAction.converse",
        "this_turn_text": "yeah, that is a lot. i like the one Bad Blood",
        "topic_constraint": "music",
        "sys_init": "not_sys_init",
        "response_candidates": [
            {
                "candidate_text": "One of the best electric guitar engineers is Leo Fender. However, he can't play the guitar himself, so he has to call musicians to test his prototypes.",
                "candidate_rg_name": "rg1",
            },
            {
                "candidate_text": "I think it is awesome how scientists have explored music as way to improve human lives.",
                "candidate_rg_name": "music_rg_2",
            },
            {
                "candidate_text": "The Japanese word 'karaoke' comes from a phrase meaning 'empty orchestra'. I love how music has roots in so many cultures.",
                "candidate_rg_name": "wiki_rg",
            },
            {
                "candidate_text": "So, what kind of music do you like?",
                "candidate_rg_name": "MUSIC",
            },
            {
                "candidate_text": "Right? This is interesting, Taylor Swift sings the song Bad Blood with Kendrick Lamar, want to hear more about Kendrick Lamar?",
                "candidate_rg_name": None
            }
        ]
    }

    for t in conv["history_turns"]:
        print(f"[USER] {t['user_text']}")
        print(f"[ATH ] {t['athena_resp']}")
    print(f"[USER] {conv['this_turn_text']}")
    print("\nScored Responses")
    for r in ranker.rank(conv):
        print("-")
        print(round(r["score"], 5), r["candidate_text"])


if __name__ == "__main__":
    main()
