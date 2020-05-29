**Последовательность запуска**

## Обработка данных

### Загрузка, обработка, разбиение данных

Полученные текстовые данные положила в папку `./files/data/`

Самый первый ноутбук, который нужно запустить `data_analyse_split.ipynb`. 
В нем происходит:
 * очистка данных
 * разбиение на train, dev, test в пропорции 60:20:20 
 * все слова переводятся в индексы и сохраняются в  `./files/data/tokens`. Максимальные индекс = |V|. V - словарь.
 * предобученный  RuBERT (iPavlov) положила в папку `./files/bert/rubert_cased_L-12_H-768_A-12_pt/`
 * все сэмплы токенизируются с помощью токенизатора RuBERT и сохраняются в папку `./files/data/bert_data`
 
### Дообучение RuBERT

 Следующий ноутбук, который нужно запустить, это `finetune_bert.ipynb`. В нем дообучается RuBERT на данных из папки `./files/data/bert_data`. Модель сохраняется в `./files/bert/rubert_finetune/`
 
### Извлечение логитов

После того, как модель обучена, извлекаем логиты. Для этого запускаем ноутбук `extract_bert_logits.ipynb`. В нем прописывается путь к данным `./files/data/bert_data` и дообученной модели RuBERT `./files/bert/rubert_finetune/`
Логиты сохраняются в папку `./files/data/logits`.

На этом заканчивается подготовка данных, и мы переходим к методу **knowledge distillation**. 

## Knowledge distillation

Нам потребуются обработанные данные `./files/data/tikens` и `./files/data/logits`.

Создаю отдельную папку с файлами:  
`./kd/  
    compress.py             
    datareader.py           
    main.py                 
    model.py`  

Все происходило на колабе. Я переместила  данные `./files/data/tikens` и `./files/data/logits` в `./kd/data`. И залила архив kd.zip в колаб. 


