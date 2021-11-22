# Загрузка необходимых пакетов

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Если в колабе, то можно просто `pip install thop`

# Файлы

1) `src` - основная директория с моделями и утилитами
2) `src/streaming.py` - стриминг
3) `Train_models.ipynb` - ноутбук с обучением всех моделей (бейзлайн + дистилляции)
4) `Results.ipynb` - ноутбук с отчетом
5) `saved` - директория с весами обученных моделей