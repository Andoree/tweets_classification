import os

import pandas as pd

def main():
    res_dir = r"test_labels/results/bilingual_pretrain_ruen/"
    output_fname = r"results_with_mean_bilingual_pretrain_ruen.csv"
    inp_fname = r"results.tsv"

    inp_path = os.path.join(res_dir, inp_fname)
    out_path = os.path.join(res_dir, output_fname)

    res = pd.read_csv(inp_path, index_col=False)
    print(res)
    mean_scores = res.mean()

    res = res.append(mean_scores, ignore_index=True, )
    print(res)
    print(mean_scores)
    #res = pd.concat((res, mean_scores), axis=0)
    res.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

