import os
import pickle
import time
import numpy as np
import tiktoken
from tinylm import PAT, Tokenizer

vocab_path = 'artifacts/tokenizer/owt_vocab.pkl'
merges_path = 'artifacts/tokenizer/owt_merges.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
special = '<|endoftext|>'
special_id = next(i for i, b in vocab.items() if b == special.encode())
ranks = {value: key for key, value in vocab.items() if key != special_id}
assert len(ranks) == len(vocab) - 1, 'duplicate token bytes'
fast = tiktoken.Encoding(name='assignment1_owt', pat_str=PAT, mergeable_ranks=ranks, special_tokens={special: special_id})
reference = Tokenizer.from_files(vocab_path, merges_path, [special])
for path in ('data/owt_train.txt', 'data/owt_valid.txt'):
    with open(path, encoding='utf-8') as source:
        for n, line in enumerate(source):
            if n >= 2000:
                break
            assert fast.encode(line, allowed_special={special}) == reference.encode(line), (path, n)
print('PASS: first 2000 lines per split match custom tokenizer exactly', flush=True)
for split in ('train', 'valid'):
    path = f'data/owt_{split}.bin.buffered.tmp'
    start = time.perf_counter()
    with open(f'data/owt_{split}.txt', encoding='utf-8') as source, open(path, 'xb') as output:
        buffer = []
        for i, line in enumerate(source, 1):
            buffer.extend(fast.encode(line, allowed_special={special}))
            if len(buffer) >= 1000000:
                output.write(np.asarray(buffer, dtype=np.uint16).tobytes())
                buffer.clear()
            if i % 100000 == 0:
                print(split, i, 'lines', output.tell(), 'bytes', flush=True)
        if buffer:
            output.write(np.asarray(buffer, dtype=np.uint16).tobytes())
    print(split, 'complete', os.path.getsize(path), 'bytes', time.perf_counter()-start, 'seconds', flush=True)
print('FAST_ENCODING_COMPLETE', flush=True)
