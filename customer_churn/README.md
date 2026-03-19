# Предсказание оттока клиентов нейронной сетью
### Бинарная классификация Kaggle - "Predict Customer Churn"

### Google Colab:
https://drive.google.com/drive/folders/1EpLYPD52Ctckrph7kkYOjtlJGLDpcQC_?usp=drive_link

### Kaggle score: 0.78535

### Параметры модели:
Сеть представленна двумя слоями нейронов на PyTorch.
Ф-ция активации: ReLU
Оптимизатор: RMSprop, lerning rate:0.01, weight_decay=0.001
Loss-функция: BCEWithLogitsLoss
Кол-во эпох: 4



Пайплайн обработки данных реализован с помощью Sklearn.

Прогресс обучения нейросети отслеживается через прогрессбар tqdm

При необходимости параметры уже обученой модели можно загрузить из файла model_dnn.tar