import codecs
import json
import numpy as np
IGNORED_TOKENS = ['[SEP]',]


def main():
    bert_features_json_path = r"../../bert/bert_features/rudrec.json"
    output_file_path = "tweets_tokens_embeddings.txt"
    with codecs.open(bert_features_json_path, "r", encoding="utf-8") as inp, \
            codecs.open(output_file_path, "w+", encoding="ascii") as output:
        for i in range(2):
            print('----' * 10)
            line = inp.readline()
            json_doc = json.loads(line)
            for k, v in json_doc.items():
                if k == "linex_index":
                    print('linex_index', v)
                # Processing of individual token
                elif k == "features":
                    num_tokens = len(v)
                    print('num_tokens', num_tokens)
                    for elem in v:
                        token = elem['token']
                        if token not in IGNORED_TOKENS:
                            token_embedding = np.zeros(shape=768, dtype=np.float32)
                            layers = elem['layers']
                            num_layers = len(layers)
                            print('num_layers', num_layers)
                            for layer in layers:
                                # single layer: dict with keys[index, embedding]
                                layer_embedding = layer['values']
                                layer_embedding = np.array(layer_embedding, dtype=np.float32)
                                token_embedding += layer_embedding
                            token_embedding /= num_layers
                            token_embedding_string = ','.join((str(x) for x in token_embedding))
                            output.write(f"{token_embedding_string}\t")
                    output.write('\n')



if __name__ == '__main__':
    main()
