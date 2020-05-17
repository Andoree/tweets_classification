import codecs
import json

def main():
    bert_features_json_path = r"../../bert/bert_features/dictionary.json"
    with codecs.open(bert_features_json_path, "r", encoding="utf-8") as inp:
        for line in inp:
            json_doc = json.load(line)
            for k, v in json_doc.items():
                print(k, type(v))


if __name__ == '__main__':
    main()

