import re
import glob
import json
import random
from tqdm import tqdm

random.seed(940803)


def cleanse_text(string):
    return re.sub(r'[\s\n]+', ' ', string).strip()


if __name__ == '__main__':

    corpora = glob.glob('E:\\Corpora & Language Resources\\모두의 말뭉치\\jsons\\*.json')

    pool = dict()
    task_id = 0
    pairs = 0
    paragraphs = 0

    train_path = 'E:/Corpora & Language Resources/모두의 말뭉치/splits/modu-paragraphs-train.txt'
    dev_path = 'E:/Corpora & Language Resources/모두의 말뭉치/splits/modu-paragraphs-dev.txt'
    train = open(train_path, encoding='utf-8', mode='w')
    dev = open(dev_path, encoding='utf-8', mode='w')

    for corpus in tqdm(corpora, desc=f'Fetching contents from {len(corpora)} files...'):
        task_id += 1
        with open(corpus, encoding='utf-8') as js:
            data = json.load(js)
        for doc in data['document']:
            targets = []
            if 'sentence' in doc.keys():
                # targets = [cleanse_text(x['form']) for x in doc['sentence']]
                pass
            elif 'paragraph' in doc.keys():
                targets = [cleanse_text(x['form']) for x in doc['paragraph']]
            elif 'utterance' in doc.keys():
                # targets = [cleanse_text(x['form']) for x in doc['utterance']]
                pass
            else:
                raise NotImplementedError
            targets = [x for x in targets if x]
            if targets:
                num_dev_samples = int(len(targets) / 10)
                random.shuffle(targets)
                paragraphs += len(targets)
                for to_train in targets[num_dev_samples:]:
                    print(to_train, file=train)
                for to_dev in targets[:num_dev_samples]:
                    print(to_dev, file=dev)

                """ for sentence pairs
                if len(targets) > 1:
                    for pair in zip(targets, targets[1:]):
                        pairs += 1
                        key = ' __split__ '.join(pair)
                        if key not in pool:
                            print(key, file=train)
                        pool[key] = None
                """
        if task_id % 1000 == 0:
            print(f'\t{paragraphs} paragraphs / {pairs} pairs fetched so far.')

    train.close()
    dev.close()
