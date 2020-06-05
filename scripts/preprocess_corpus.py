import codecs
import os
import re

import chardet
import nltk
from nltk.tokenize import sent_tokenize

NAME_DICT_STR_BEGI_PATTERN = r"^[0-9 ._№]"


def list_replace(search, replacement, text):
    """
    Заменяет множество символов строки заданным
        :param search: строка заменяемых символов
        :param replacement: строка, которой заменяются заданные
        :param text: строка, в которой происходит поиск и замена
        :return: результат замены (строка)
    """
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text


# принимает строку в юникоде
def unify_sym(text):
    # Унифицирование кавычек
    text = list_replace \
        ('\u00AB\u00BB\u2039\u203A\u201E\u201A\u201C\u201F\u2018\u201B\u201D\u2019', '\u0022', text)
    # Унифицирование тире
    text = list_replace \
        ('\u2012\u2013\u2014\u2015\u203E\u0305\u00AF', '\u2003\u002D\u002D\u2003', text)
    # Унифицирование дефисов и "минусов"
    text = list_replace('\u2010\u2011', '\u002D', text)
    # Унифицирование пробелов
    text = list_replace \
            (
            '\u2000\u2001\u2002\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u2060\u3000',
            '\u2002', text)

    text = re.sub('\u2003\u2003', '\u2003', text)
    text = re.sub('\t\t', '\t', text)

    # Унифицирование точек
    text = list_replace \
            (
            '\u02CC\u0307\u0323\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u00B7\u00D7\u22C5\u2219\u2062',
            '.', text)

    # Унифицирование "звёздочек"
    text = list_replace('\u2217', '\u002A', text)

    text = list_replace('…', '...', text)

    # Удаление диерезисов над латинскими буквами
    text = list_replace('\u00C4', 'A', text)
    text = list_replace('\u00E4', 'a', text)
    text = list_replace('\u00CB', 'E', text)
    text = list_replace('\u00EB', 'e', text)
    text = list_replace('\u1E26', 'H', text)
    text = list_replace('\u1E27', 'h', text)
    text = list_replace('\u00CF', 'I', text)
    text = list_replace('\u00EF', 'i', text)
    text = list_replace('\u00D6', 'O', text)
    text = list_replace('\u00F6', 'o', text)
    text = list_replace('\u00DC', 'U', text)
    text = list_replace('\u00FC', 'u', text)
    text = list_replace('\u0178', 'Y', text)
    text = list_replace('\u00FF', 'y', text)
    text = list_replace('\u00DF', 's', text)
    text = list_replace('\u1E9E', 'S', text)

    currencies = list \
            (
            '\u20BD\u0024\u00A3\u20A4\u20AC\u20AA\u2133\u20BE\u00A2\u058F\u0BF9\u20BC\u20A1\u20A0\u20B4\u20A7\u20B0\u20BF\u20A3\u060B\u0E3F\u20A9\u20B4\u20B2\u0192\u20AB\u00A5\u20AD\u20A1\u20BA\u20A6\u20B1\uFDFC\u17DB\u20B9\u20A8\u20B5\u09F3\u20B8\u20AE\u0192'
        )

    alphabet = list \
            (
            '\t\n\r абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ,.[]{}()=+-−*&^%$#@!~;:0123456789§/\|"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    alphabet.append("'")

    allowed = set(currencies + alphabet)

    cleaned_text = [sym for sym in text if sym in allowed]
    cleaned_text = ''.join(cleaned_text)

    return cleaned_text


def process_txt_file(path, output_file):
    with open(path, 'rb') as f:
        rawdata = b''.join([f.readline() for _ in range(100)])
        result = chardet.detect(rawdata)
        encoding = result['encoding']
    try:
        with codecs.open(path, "r", encoding=encoding) as inp_file:
            for line in inp_file:
                line = line.strip()
                line = unify_sym(line)
                if line != '':
                    sentences = sent_tokenize(line, language="russian")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        output_file.write(f"{sentence}\n")
            output_file.write("\n")
    except Exception:
        print(path)


def traverse_directory(base_dir, output_file):
    for dirpath, dirs, files in os.walk(os.path.join(base_dir)):
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            if filename != 'name_dict.txt':
                process_txt_file(file_path, output_file)


def main():
    nltk.download('punkt')
    base_dir = r"MED_txt"
    output_dir = "preprocessed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_fname = "bert_pretr_corpus.txt"
    output_path = os.path.join(output_dir, output_fname)
    with codecs.open(output_path, "w+", encoding="utf-8") as output_file:
        traverse_directory(base_dir, output_file)


if __name__ == '__main__':
    main()
