import codecs
import json

def main():
    bert_features_json_path = r"../../bert/bert_features/dictionary.json"
    with codecs.open(bert_features_json_path, "r",) as inp:
        for i in range(10):
            line = inp.readline()
            json_doc = json.loads(line)
            for k, v in json_doc.items():
                print(i, k,type( v))
                if k == "linex_index":
                    print(v)
                elif k == "features":
                    for elem in v:
                        # dictionary with keys [token, layers]
                        print('aa', elem.keys())
                        layers = elem['layers']
			# list of layers
                        print("layers", type(layers))
                        for j in layers:
                            # single layer: dict with keys[index, embedding]
                            print("bbbbbbb", j.keys())
                            print(j['index'])
                            print(len(j['values']))



if __name__ == '__main__':
    main()

