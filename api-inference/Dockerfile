# Используем легковесный официальный образ Python
FROM python:3.12-slim

# Настраиваем рабочую директорию внутри контейнера
WORKDIR /workspace

# Запрещаем Python писать файлы .pyc на диск (оптимизация памяти)
ENV PYTHONDONTWRITEBYTECODE=1
# Запрещаем Python буферизировать логи (чтобы сразу видеть их в docker logs)
ENV PYTHONUNBUFFERED=1

# Копируем файл зависимостей
COPY requirements.txt .

# Обновляем pip и устанавливаем библиотеки без сохранения кэша (уменьшает размер образа)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код проекта в контейнер
COPY . .

# Открываем порт 8000, на котором будет крутиться FastAPI
EXPOSE 8000

# Команда для запуска приложения при старте контейнера.
# Важно: host ставим 0.0.0.0, чтобы принимать запросы извне контейнера!
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]