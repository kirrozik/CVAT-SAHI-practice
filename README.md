# Летняя практика: Интеграция YOLO11x и SAHI в CVAT

Данный репозиторий содержит результаты летней практики по интеграции кастомных моделей детекции объектов в CVAT (Computer Vision Annotation Tool) с использованием serverless-функций Nuclio.

## Выполненные задания

- **Задание 0:** Развертывание CVAT с компонентом Serverless (Nuclio).
- **Задание 1:** Запуск дефолтной модели YOLOv7 (ONNX).
- **Задание 2:** Разработка Nuclio-функции для **YOLO11x** (библиотека `ultralytics`).
- **Задание 3:** Интеграция **SAHI (Slicing Aided Hyper Inference)** с YOLO11x для детекции мелких объектов.

## Требования к окружению (Linux)

- Дистрибутив на базе Ubuntu/Debian
- Docker Engine и Docker Compose Plugin
- Git
- Не менее 8 ГБ оперативной памяти (для работы YOLO11x и SAHI)

## Пошаговая инструкция по запуску

### 1. Клонирование и настройка CVAT

```bash
git clone https://github.com/opencv/cvat
cd cvat
git checkout v2.68.0
```

Скопируйте файл `docker-compose.override.yml` из этого репозитория в корень склонированной папки `cvat`. Этот файл необходим для корректной работы сети между контейнерами CVAT и Nuclio через `host.docker.internal` в нативном Linux.

Запустите CVAT:

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml -f components/serverless/docker-compose.serverless.yml up -d
```

### 2. Создание администратора и установка nuctl

```bash
docker exec -it cvat_server python3 manage.py createsuperuser

wget https://github.com/nuclio/nuclio/releases/download/1.15.9/nuctl-1.15.9-linux-amd64
chmod +x nuctl-1.15.9-linux-amd64
sudo mv nuctl-1.15.9-linux-amd64 /usr/local/bin/nuctl
nuctl create project cvat --platform local
```

### 3. Деплой моделей

**Деплой YOLOv7 (дефолтная модель из репозитория CVAT):**

```bash
cd serverless/onnx/WongKinYiu/yolov7/nuclio
nuctl deploy --project-name cvat --path . --file function.yaml --platform local \\
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \\
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \\
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
cd ~/cvat
```

**Деплой кастомной YOLO11x:**

Скопируйте папку `yolo11x` из этого репозитория в удобное место и выполните:

```bash
cd /путь/к/папке/yolo11x
nuctl deploy --project-name cvat --path . --file function.yaml --platform local \\
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \\
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \\
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

**Деплой YOLO11x + SAHI:**

Скопируйте папку `yolo11x_sahi` и выполните:

```bash
cd /путь/к/папке/yolo11x_sahi
nuctl deploy --project-name cvat --path . --file function.yaml --platform local \\
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \\
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \\
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

*Примечание: Первый деплой моделей YOLO11x займет 5-10 минут, так как контейнер будет устанавливать CPU-версию PyTorch и скачивать веса моделей (~140 МБ).*

### 4. Тестирование в CVAT

1. Откройте `http://localhost:8080` и войдите в систему.
2. Создайте задачу (Task), добавьте лейблы из датасета COCO (например, `person`, `car`).
3. В меню **Actions -> Automatic annotation** выберите модель `YOLO 11x` или `YOLO 11x (SAHI)`.
4. Свяжите ваши лейблы с классами нейросети (Label mapping) и нажмите **Annotate**.

## Архитектурные заметки

- Для избежания скачивания тяжелой CUDA-версии PyTorch (~2.5 ГБ) в `function.yaml` используется установка CPU-версии через официальный индекс wheel.
- Флаг `--platform-config` с сетью `cvat_cvat` обязателен для доступа функции к внутреннему Redis CVAT.
- В коде SAHI реализован автоматический даунскейл изображений свыше 1920p и конвертация в RGB для предотвращения ошибок обработки альфа-каналов и таймаутов CVAT (120 сек).
- Файл `docker-compose.override.yml` решает проблему маршрутизации `host.docker.internal` в нативном Linux.
