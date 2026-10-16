# OCR чат (Android)

Чат-бот офлайн-OCR: отправляешь картинку — бот отвечает текстом строк в том же
месте и порядке, плюс аннотированная копия картинки с боксами и подписями.
Работает **без настроек**: модель r4 встроена, любую свою `.onnx` (CRNN+CTC,
blank=109/unk=110, вход 1×1×32×W) можно загрузить кнопкой 📦 прямо из файловой
системы.

## Возможности
- 📎 chat-UI (RecyclerView): баблы пользователя — картинки, баблы бота — текст
- каждая найденная строка: координаты `(x,y)`, текст, confidence (CTC softmax-медиана)
- аннотированная копия картинки: боксы + текст прямо на месте
- ошибки приходят в чат **полностью со stacktrace**
- все координатные кропы строго клампятся в рамки битмапа (защита от `empty range / max<min`)
- никаких regex над OCR-выводом

## Пайплайн (порт OnnxRu из Yomikai)
1. страница даунскейлится до 1400 px по длинной стороне
2. luma < 170 → морфозамыкание 25×3 (×2) → связные компоненты (h≥10, w≥12)
3. группировка компонент в строки по позиции центра
4. readLine: высота 32 px, белый правый пад до кратного 8, luma/255
5. CTC greedy-декодер (blank=109, unk=110), conf = медиана softmax вероятностей

## Сборка
```bash
export JAVA_HOME=<JDK17>
echo "sdk.dir=<путь к android-sdk>" > local.properties
gradle assembleRelease        # производит app-release-unsigned.apk
apksigner sign --ks <keystore> ...
```
Зависимость движка: `com.microsoft.onnxruntime:onnxruntime-android:1.17.3`
(arm64-v8a + armeabi-v7a, minSdk 24, targetSdk 34).

## Структура
- `app/` — весь исходный код (Kotlin, 3 файла: MainActivity, ChatAdapter, OcrEngine)
  + ресурсы, Manifest и встроенная модель `app/src/main/assets/best.onnx`
- `models/` — все версии моделей (baseline / r3 / r4)
- `releases/` — подписанный `OCR-chat.apk` (~34 МБ)

## Модель
CRNN, charset 111 символов (кириллица RU+UK, цифры, пунктуация).
Требования к подменяемой модели: вход `1×1×32×W` float32 (luma/255), выход `1×T×C`, CTC.
