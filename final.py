
import timm 
timm.list_models(pretrained=True)
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score,f1_score
import itertools
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
#data_dir = '/content/drive/MyDrive/Colab Notebooks/big data'
data_dir=os.getcwd()
model_list=['efficientnet_b0','xception']
high='/2018313925'
low='/2018313925'
valid_ratio = 0.5  # 검증 데이터셋의 비율
test_ratio = 0.5 
df = pd.read_csv(data_dir+'/2018313925/your_prediction.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_list=[high,low]
accuracy_list=[]
class CustomDataset(Dataset):
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []  
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(self.root_dir, filename)
                self.file_paths.append(file_path)
  
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(0) if df["label"].iloc[0]==0 else torch.tensor(1)
        if self.transform:
            image = self.transform(image)
        return image.to(device), label.to(device)
    
    def load_image(self, file_path):
      pass
#앙상블
class EnsembleModel(nn.Module):
    def __init__(self, model_names, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([timm.create_model(name, pretrained=True) for name in model_names])
        
        self.classifier = nn.Linear(len(model_names) * self.models[0].num_classes, num_classes)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(0) if df["label"].iloc[0]==0 else torch.tensor(1)
        if self.transform:
            image = self.transform(image)
        return image.to(device), label.to(device)
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.cat(outputs, dim=1)
        logits = self.classifier(outputs)
        return logits
num_classes = 2
ensemble_model = EnsembleModel(model_list, num_classes)
ensemble_model.to(device)
y_true_train, y_pred_train = [], []
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # (224,224)크기
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화 과정
])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)
num_epochs = 10

for i in [x for x in range(0, 20) if x != 9]:
    train_dataset = CustomDataset(root_dir=data_dir +high +'/video_'+str(i), transform=transform)
    train_data, valid_data = train_test_split(train_dataset, test_size=valid_ratio, shuffle=True, random_state=42)
    test_data=valid_data
    #data 로더
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        ensemble_model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = ensemble_model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        ensemble_model.eval()
        correct = 0
        total = 0

    with torch.no_grad():
        for images, labels in train_dataloader:
            
            images = images.to(device)
            labels = labels.to(device)

            logits = ensemble_model(images)
            _, predicted = torch.max(logits.data, 1)
            prob = F.softmax(logits, dim=1)
            y_pred_train+=[round(prob, 2) for prob in prob[:, 1].tolist()]
            # mean_prob=np.sum(prob[:, 1].tolist())/2
            # y_pred_train+=[round(mean_prob, 2) ]
            
print(y_pred_train)
# 결과 기록
df['Ensemble'] = y_pred_train
df.to_csv(data_dir + '/2018313925/your_prediction.csv', index=False)



#hq
# for i in [x for x in range(0, 20) if x != 9]:
#   train_dataset = CustomDataset(root_dir=data_dir +high +'/video_'+str(i), transform=transform)
#   train_data, valid_data = train_test_split(train_dataset, test_size=valid_ratio, shuffle=True, random_state=42)
#   test_data=valid_data
#   #data 로더
#   train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
#   valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
#   test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   model = timm.create_model(model_list[0],num_classes=2, pretrained=True).to(device)
#   criterion = nn.CrossEntropyLoss()
#   optimizer = optim.Adam(model.parameters(), lr=0.001)
#   train_losses = []
#   valid_losses = []
  
#   epochs = 5

#   for epoch in range(epochs):
#       model.train()
#       train_loss = 0.0
#       valid_loss = 0.0
#       total_samples = 0
#       correct_predictions = 0
      
#       for images, labels in train_dataloader:
#           images = images.to(device)
#           labels = labels.to(device)
          
#           optimizer.zero_grad()
          
#           outputs = model(images)
#           loss = criterion(outputs, labels)
          
#           loss.backward()
#           optimizer.step()
          
#           train_loss += loss.item() * images.size(0)
#           _, predicted = torch.max(outputs.data, 1)
#           total_samples += labels.size(0)
#           correct_predictions += (predicted == labels).sum().item()
      
#       train_loss = train_loss / total_samples
#       train_accuracy = correct_predictions / total_samples
#       print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
#       train_losses.append(train_loss)
#       model.eval()
#       correct = 0
#       total = 0

#       with torch.no_grad():
#           for images, labels in valid_dataloader:
#               images = images.to(device)
#               labels = labels.to(device)

#               outputs = model(images)
#               _, predicted = torch.max(outputs.data, 1)
#               valid_loss += loss.item() * images.size(0)
#               total += labels.size(0)
#               correct += (predicted == labels).sum().item()

#       accuracy = correct / total
      
#       valid_loss=valid_loss/total
#       valid_losses.append(valid_loss)


#   model.eval()
#   correct = 0
#   total = 0

#   with torch.no_grad():
#       for images, labels in test_dataloader:
#           images = images.to(device)
#           labels = labels.to(device)

#           outputs = model(images)
#           _, predicted = torch.max(outputs.data, 1)

#           total += labels.size(0)
#           correct += (predicted == labels).sum().item()

#   accuracy = correct / total
#   accuracy_list.append(accuracy)
 
#   print(f'Epoch 100, Validation Accuracy: {accuracy:.4f}')
  
#   with torch.no_grad():
#       for images, labels in train_dataloader:
#           images = images.to(device)
#           labels = labels.to(device)

#           outputs = model(images)
#           prob = F.softmax(outputs, dim=1)
#           _, predicted = torch.max(outputs.data, 1)

#           y_true_train += labels.tolist()
        
#           y_pred_train+=[round(prob, 2) for prob in prob[:, 1].tolist()]
          
          
          

# df['Hq'] = y_pred_train
# df.to_csv(data_dir+'/2018313925/your_prediction.csv', index=False)
# print(np.sum(accuracy_list)/20)




