import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, vectorizer=None, fit_vectorizer=False):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer

        # 特徴量エンジニアリング
        self.df['num_characters_question'] = self.df['question'].apply(lambda x: len(x))
        self.df['num_words_question'] = self.df['question'].apply(lambda x: len(x.split(' ')))
        self.df['num_sentences_question'] = self.df['question'].apply(lambda x: len(x.split('.')))

        # TF-IDFの計算
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None, analyzer='word', ngram_range=(1, 2))
        else:
            self.vectorizer = vectorizer

        if fit_vectorizer:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['question'])
        else:
            self.tfidf_matrix = self.vectorizer.transform(self.df['question'])

        self.tfidf_df = pandas.DataFrame(self.tfidf_matrix.toarray())
        self.df = pandas.concat([self.df.reset_index(drop=True), self.tfidf_df.reset_index(drop=True)], axis=1)

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question = np.zeros(len(self.idx2question) + 1)
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1
            except KeyError:
                question[-1] = 1

        # 追加した特徴量を取得
        num_characters_question = self.df['num_characters_question'][idx]
        num_words_question = self.df['num_words_question'][idx]
        num_sentences_question = self.df['num_sentences_question'][idx]
        additional_features = torch.tensor([num_characters_question, num_words_question, num_sentences_question], dtype=torch.float32)

        # TF-IDF特徴量を取得
        tfidf_features = torch.tensor(self.tfidf_matrix[idx].toarray(), dtype=torch.float32).squeeze()

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx), additional_features, tfidf_features
        else:
            return image, torch.Tensor(question), additional_features, tfidf_features

    def __len__(self):
        return len(self.df)



# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, additional_feature_size: int = 3, tfidf_feature_size: int = 100):
        super().__init__()
        self.resnet = ResNet50()
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024 + additional_feature_size + tfidf_feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question, additional_features, tfidf_features):
        image_feature = self.resnet(image)
        question_feature = self.text_encoder(question)
        
        # 追加した特徴量を結合
        x = torch.cat([image_feature, question_feature, additional_features, tfidf_features], dim=1)
        x = self.fc(x)

        return x



# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer, additional_features, tfidf_features in dataloader:
        image, question, answers, mode_answer, additional_features, tfidf_features = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), additional_features.to(device), tfidf_features.to(device)

        pred = model(image, question, additional_features, tfidf_features)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer, additional_features, tfidf_features in dataloader:
        image, question, answers, mode_answer, additional_features, tfidf_features = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), additional_features.to(device), tfidf_features.to(device)

        pred = model(image, question, additional_features, tfidf_features)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None, analyzer='word', ngram_range=(1, 2))

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform, vectorizer=vectorizer, fit_vectorizer=True)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False, vectorizer=vectorizer)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx), tfidf_feature_size=train_dataset.tfidf_df.shape[1]).to(device)

    num_epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    model.eval()
    submission = []
    for image, question, additional_features, tfidf_features in test_loader:
        image, question, additional_features, tfidf_features = image.to(device), question.to(device), additional_features.to(device), tfidf_features.to(device)
        pred = model(image, question, additional_features, tfidf_features)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission_3.npy", submission)

if __name__ == "__main__":
    main()

    
# 特徴量エンジニアリング　質問の文字数などの追加
# rotationなど
# TF-IDFを追加
# resnet50
'''
train time: 1422.70 [s]
train loss: 6.2339
train acc: 0.4692
train simple acc: 0.3781
'''