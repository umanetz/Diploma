
### Обучение

По дефолту размер скрытого слоя 1000.


Обучение может происходить для двух типов моделей -`-model_type`: `stack_biLSTM` и `stack_LSTM`. Модель сохраняется в папку вида `./2020-06-01___22-48-13/model_distill_[model_type]`, где название папки - это время начала тренировки. 
Обязательно задавать путь к данным   `--data_path` и тип модели `--model_type`. Доступна регулировка других параметров. Подробнее в `main.py`

Запуск:  
`python main.py --data_path ./data/ --model_type [model_type]` 



### Визуализация и сжатие

Для сжатия модели запускается скрипт `compress.py` с указанием пути к обученной модели `--restore_path ./2020-06-01___22-48-13/`. Мы можем визуализировать матрицы весов (флаг `--show_weight`) или/и создать новую модель с удалением ненужных компонент (флаг `--cut_model`). Сжатая модель и графики весов сохраняются в `--restore_path`

Запуск, чтобы визуализировать веса и сохранить сжатую модель:

`python compress.py --restore_path ./2020-06-01___22-48-13  --show_weight  --cut_model` 



Чтобы оценить качество сжатой модели, убедиться, что все прошло хорошо, снова используем `main.py`.  Загружаем модель из `--restore_model_path ./2020-06-01___22-48-13/cut_model_distill_[model_type]`, обязательно указывая путь к данным `--data_path` и тип модели `--model_type` как при обучении: 

`python main.py --data_path ./data/ --restore_model_path ./2020-06-01___22-48-13/cut_model_distill_[model_type].pt --model_type [model_type]`
