import codecs
import json

def main():
    bert_features_json_path = r"../../bert/bert_features/rudrec.json"
    with codecs.open(bert_features_json_path, "r",encoding="utf-8") as inp:
        for i in range(2):
            print('----' * 10)
            line = inp.readline()
            json_doc = json.loads(line)
            for k, v in json_doc.items():
                # print(i, k,type( v))
                #print('--' * 10)
                if k == "linex_index":
                    print(v)
                elif k == "features":
                    for elem in v:
                        token = elem['token']
                        #if token == '[CLS]':
                            #print(token)
                        # dictionary with keys [token, layers]
                        elem['token']
                        print('\taa', elem.keys())
                        layers = elem['layers']
			# list of layers
                        print('\telem.token', elem['token'])
                        print("\tlayers", type(layers))
                        for j in layers:
                            # single layer: dict with keys[index, embedding]
                            print("\t\tbbbbbbb", j.keys())
                            print('\t\tIndex',j['index'])
                            print( '\t\t len vectors',len(j['values']))



if __name__ == '__main__':
    main()

