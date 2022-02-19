import time
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

start = time.time()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(files=["E:\Corpora & Language Resources\모두의 말뭉치\splits\modu-plm-test.txt",
                       "E:\Corpora & Language Resources\모두의 말뭉치\splits\modu-plm-dev.txt",
                       "E:\Corpora & Language Resources\모두의 말뭉치\splits\modu-plm-train.txt"], trainer=trainer)

print('runtime:', time.time() - start)

print(tokenizer.get_vocab_size())
print(tokenizer.get_vocab())

tokenizer.save('huggingface.json')
