import codecs
import json
import numpy as np

def main():
    bert_features_json_path = r"../../bert/bert_features/rufren_tweets_embeddings.json"
    output_file_path = "../tweets_embeddings/rufren_cls_embs.txt"
    with codecs.open(bert_features_json_path, "r", encoding="utf-8") as inp, \
            codecs.open(output_file_path, "w+", encoding="ascii") as output:
        for i, line in enumerate(inp):
            print(i)
            json_doc = json.loads(line)
            for k, v in json_doc.items():
                if k == "linex_index":
                    pass
                # Processing of individual token
                elif k == "features":
                    num_tokens = len(v)
                    for elem in v:
                        token = elem['token']
                        if token == '[CLS]':
                            token_embedding = np.zeros(shape=768, dtype=np.float32)
                            layers = elem['layers']
                            num_layers = len(layers)
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
