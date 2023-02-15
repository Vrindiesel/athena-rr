# Athena-RR
A Transformer-based Response Evaluator for Open-Domain Spoken Conversation

```python
from response_ranker import ResponseRanker
ranker = ResponseRanker('../model')

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
```


```commandline
$ python response_ranker.py ../model
args: Namespace(checkpoint_path='../model')
loading: ../model/tokenizer.pt
loading model path: ../model
[USER] let's talk about music
[ATH ] When I listen to music the musician <prosody pitch='+15%' rate='85%' volume='loud'> Taylor Swift </prosody> is one of my favorites. Do you have a favorite musician?
[USER] yeah, taylor swift.
[ATH ] Ok, Taylor Swift. Wow! Taylor Swift is very prolific! She has 114 songs, that’s a lot!
[USER] yeah, that is a lot. i like the one Bad Blood

Scored Responses
-
0.0006 One of the best electric guitar engineers is Leo Fender. However, he can't play the guitar himself, so he has to call musicians to test his prototypes.
-
0.00063 I think it is awesome how scientists have explored music as way to improve human lives.
-
0.0006 The Japanese word 'karaoke' comes from a phrase meaning 'empty orchestra'. I love how music has roots in so many cultures.
-
0.00476 So, what kind of music do you like?
-
0.87039 Right? This is interesting, Taylor Swift sings the song Bad Blood with Kendrick Lamar, want to hear more about Kendrick Lamar?

```
