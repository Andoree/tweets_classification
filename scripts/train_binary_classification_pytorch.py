import codecs
import configparser
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available else "cpu"


class TweetsDataset(Dataset):
    def __init__(self, tweets_df, text_tokenizer, max_length=128):
        self.labels = tweets_df["class"].astype(np.float32).values
        self.max_length = max_length
        self.tokenized_tweets = [text_tokenizer.encode_plus(x, max_length=self.max_length,
                                                            padding="max_length", truncation=True,
                                                            return_tensors="pt", ) for x in tweets_df.tweet.values]
        self.drug_embeddings = tweets_df.drug_embedding.values

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_tweets[idx]["input_ids"][0],
            "attention_mask": self.tokenized_tweets[idx]["attention_mask"][0],
            "drug_embeddings": self.drug_embeddings[idx],
            "labels": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def create_dataset_weights(dataset):
    count_dict = {}
    for item in dataset:
        label = item["labels"]
        if count_dict.get(label) is None:
            count_dict[label] = 0
        count_dict[label] += 1
    num_samples = len(dataset)
    label_to_weight = {}
    assert num_samples == sum(count_dict.values())
    for cl, count in count_dict.items():
        freq = count / num_samples
        label_to_weight[cl] = 1 - freq
    sample_weights = np.empty(num_samples, dtype=np.float)
    for i, item in enumerate(dataset):
        label = item["labels"]
        sample_weights[i] = label_to_weight[label]
    return sample_weights


def train(model, iterator, optimizer, criterion, train_history=None, valid_history=None, use_drug_embeddings=True):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if use_drug_embeddings:
            drug_embeddings = batch["drug_embeddings"].to(device)
            output = model(inputs=input_ids, attention_mask=attention_mask, drug_embeddings=drug_embeddings).squeeze(1)
        else:
            output = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())

    return epoch_loss / (i + 1)


def encode_smiles(model, tokenizer, smiles_list, max_length, molecules_sep='~~~'):
    model.eval()
    with torch.no_grad():
        model_hidden_size = model.config.hidden_size
        molecules_embeddings = []
        for sample in tqdm(smiles_list):
            sample_embeddings = []
            if sample is not np.nan:
                molecules_smiles = sample.split(molecules_sep)
                for smile_str in molecules_smiles:
                    encoded_molecule = tokenizer.encode(smile_str, max_length=max_length,
                                                        padding="max_length", truncation=True, return_tensors="pt").to(
                        device)
                    output = model(encoded_molecule, return_dict=True)
                    cls_embedding = output["last_hidden_state"][0][0].cpu()
                    sample_embeddings.append(cls_embedding)
                mean_sample_embedding = torch.mean(torch.stack(sample_embeddings), dim=0)
            else:
                mean_sample_embedding = torch.zeros(size=[model_hidden_size, ], dtype=torch.float32)
            molecules_embeddings.append(mean_sample_embedding)
    return molecules_embeddings


def evaluate(model, iterator, criterion, use_drug_embeddings):
    model.eval()

    epoch_loss = 0

    true_labels = []
    pred_labels = []

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            true_labels.extend(labels.cpu().numpy())
            labels = labels.to(device)

            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                output = model(inputs=input_ids, attention_mask=attention_mask,
                               drug_embeddings=drug_embeddings).squeeze(1)
            else:
                output = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)
            pred_probas = output.cpu().numpy()
            batch_pred_labels = (pred_probas >= 0.5) * 1

            loss = criterion(output, labels)

            pred_labels.extend(batch_pred_labels)
            epoch_loss += loss.item()

    valid_f1_score = f1_score(true_labels, pred_labels)
    return epoch_loss / (i + 1), valid_f1_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, n_epochs, use_drug_embeddings,
                   save_checkpoint_path, output_evaluation_path):
    train_history = []
    valid_history = []
    valid_history_f1 = []

    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = -1

    eval_dir = os.path.dirname(output_evaluation_path)
    train_statistics_path = os.path.join(eval_dir, "training_logs.txt")

    for epoch in tqdm(range(n_epochs)):

        start_time = time.time()

        train_loss = train(bert_classifier, train_loader, optimizer, criterion, train_history, valid_history,
                           use_drug_embeddings)
        valid_loss, valid_f1_score = evaluate(bert_classifier, dev_loader, criterion, use_drug_embeddings)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        valid_history_f1.append(valid_f1_score)

        if valid_f1_score > best_f1_score:
            best_f1_score = valid_f1_score
            best_epoch = epoch
            torch.save(bert_classifier.state_dict(), save_checkpoint_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}')

        with codecs.open(train_statistics_path, 'a+', encoding="utf-8") as output_path:
            output_path.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
            output_path.write(f'\tTrain Loss: {train_loss:.3f}\n')
            output_path.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}\n')

    return best_epoch


def train_evaluate_model(seed, bert_classifier, use_drug_embeddings, learning_rate, train_loader, dev_loader,
                         test_loader, num_epochs, output_evaluation_path, output_model_dir, model_chkpnt_name):
    torch.manual_seed(seed)
    enrudr_model = AutoModel.from_pretrained("cimm-kzn/enrudr-bert")
    optimizer = optim.Adam(bert_classifier.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    output_ckpt_path = os.path.join(output_model_dir, f"best-val-{model_chkpnt_name}.pt")
    best_epoch = train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, num_epochs,
                                use_drug_embeddings,
                                output_ckpt_path, output_evaluation_path)

    bert_classifier.load_state_dict(torch.load(output_ckpt_path))

    true_labels, pred_labels = predict(bert_classifier, dev_loader, use_drug_embeddings)
    val_model_precision = precision_score(true_labels, pred_labels)
    val_model_recall = recall_score(true_labels, pred_labels)
    val_model_f1 = f1_score(true_labels, pred_labels)

    true_labels, pred_labels = predict(bert_classifier, test_loader, use_drug_embeddings)
    test_model_precision = precision_score(true_labels, pred_labels)
    test_model_recall = recall_score(true_labels, pred_labels)
    test_model_f1 = f1_score(true_labels, pred_labels)

    with codecs.open(output_evaluation_path, 'a+', encoding="utf-8") as output_file:
        output_file.write(f"{model_chkpnt_name},{best_epoch},{val_model_precision},{val_model_recall},{val_model_f1}\n")
        output_file.write(
            f"{model_chkpnt_name},{best_epoch},{test_model_precision},{test_model_recall},{test_model_f1}\n")

    del optimizer
    del criterion


def predict(model, data_loader, use_drug_embeddings):
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_true_labels = batch["labels"].cpu().numpy()

            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                pred_probas = model(inputs=input_ids, attention_mask=attention_mask,
                                    drug_embeddings=drug_embeddings).squeeze(1)
            else:
                pred_probas = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)

            pred_probas = pred_probas.cpu().numpy()

            batch_pred_labels = (pred_probas >= 0.5) * 1

            pred_labels.extend(batch_pred_labels)
            true_labels.extend(batch_true_labels)
    return true_labels, pred_labels


class BertSimpleClassifier(nn.Module):
    def __init__(self, bert_text_encoder, dropout):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        # self.dropout = nn.Dropout(dropout)
        bert_hidden_dim = bert_text_encoder.config.hidden_size

        self.classifier = nn.Sequential(

            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(100),
            # nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, ):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        # text_cls_embeddings = self.dropout(text_cls_embeddings)

        proba = self.classifier(text_cls_embeddings)
        return proba




def clear():
    os.system('cls')


def embedding_str_to_numpy(s):
    numbers_strs = s.strip("[]").split()
    emb_size = len(numbers_strs)
    embedding = np.empty(shape=emb_size, dtype=np.float)
    for i in range(emb_size):
        embedding[i] = np.float(numbers_strs[i])
    return embedding


def main():
    config = configparser.ConfigParser()
    config.read("train_config.ini")
    model_dir = config["INPUT"]["MODEL_DIR"]
    data_dir = config["INPUT"]["INPUT_DIR"]
    seed = config.getint("PARAMETERS", "SEED")
    max_length = config.getint("PARAMETERS", "MAX_TEXT_LENGTH")
    batch_size = config.getint("PARAMETERS", "BATCH_SIZE")
    learning_rate = config.getfloat("PARAMETERS", "LEARNING_RATE")
    num_epochs = config.getint("PARAMETERS", "NUM_EPOCHS")
    apply_upsampling = config.getboolean("PARAMETERS", "APPLY_UPSAMPLING")
    model_type = config["PARAMETERS"]["MODEL_TYPE"]
    output_dir = config["OUTPUT"]["OUTPUT_DIR"]
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_evaluation_filename = config["OUTPUT"]["EVALUATION_FILENAME"]
    output_evaluation_path = os.path.join(output_dir, output_evaluation_filename)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    dev_path = os.path.join(data_dir, "dev.csv")

    train_df = pd.read_csv(train_path, )
    dev_df = pd.read_csv(dev_path, )
    test_df = pd.read_csv(test_path, )


    text_tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="models/")

    train_tweets_dataset = TweetsDataset(train_df, text_tokenizer, max_length=max_length, )
    dev_tweets_dataset = TweetsDataset(dev_df, text_tokenizer, max_length=max_length)
    test_tweets_dataset = TweetsDataset(test_df, text_tokenizer, max_length=max_length)

    russian_train_weights = create_dataset_weights(train_tweets_dataset)
    russian_train_weights = torch.DoubleTensor(russian_train_weights)

    if apply_upsampling:
        russian_sampler = torch.utils.data.sampler.WeightedRandomSampler(russian_train_weights,
                                                                         len(russian_train_weights))
        shuffle = False
    else:
        russian_sampler = None
        shuffle = True

    num_workers = 4

    train_loader = torch.utils.data.DataLoader(
        train_tweets_dataset, batch_size=batch_size, num_workers=num_workers, sampler=russian_sampler, shuffle=shuffle,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )

    DROPOUT = 0.2
    if model_type == "ru_nodrug":
        enrudr_model = AutoModel.from_pretrained(model_dir, cache_dir="models/")
        use_drug_embeddings = False
        bert_simple_clf = BertSimpleClassifier(enrudr_model, dropout=DROPOUT).to(device)
        train_evaluate_model(seed, bert_simple_clf, use_drug_embeddings, learning_rate, train_loader, dev_loader,
                             test_loader, num_epochs, output_evaluation_path, output_dir, "ru-simple")
        del bert_simple_clf
        del enrudr_model



if __name__ == '__main__':
    main()
