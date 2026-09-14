# Real3D — захват экрана/окна → стерео 3D (SBS) для XREAL на Apple Silicon

Нативное macOS-приложение (Swift + Metal + Core ML, без Python в рантайме). Захватывает дисплей
или окно через ScreenCaptureKit, считает карту глубины монокулярной моделью (Depth Anything V2
Small, 16:9-сборки) и синтезирует стереопару 2×1920×1080 (Full SBS 3840×1080) для очков.

Целевое железо: MacBook Pro M4 Max (40 GPU), 48 ГБ. macOS 14+.

## Что внутри

| Слой | Технология | Стоимость на кадр (M4 Max) |
|---|---|---|
| Захват | ScreenCaptureKit, BGRA IOSurface @60 fps, без копий (CVMetalTextureCache) | ~0 |
| Кроп 16:9 + масштаб в 1080p | Metal compute, суперсэмплинг при даунскейле, **MetalFX Spatial** при апскейле (окно меньше 1080p) | ~0.3 мс |
| Глубина | Core ML, `DAv2-Small-Balanced-644x364` по умолчанию, GPU | **14.7 мс → ~65 fps** |
| Нормализация + временное сглаживание (EMA) | Metal compute | ~0.05 мс |
| Апсемплинг глубины 644×364 → 1920×1080 | Joint-bilateral (edge-aware по цвету) | ~0.5 мс |
| Стерео-синтез | Scanline DIBR: поиск по строке, окклюзии «ближний побеждает», заполнение дырок фоном | ~1 мс |
| Вывод | CAMetalLayer, CVDisplayLink на дисплее очков | ~0.1 мс |

Полный Metal-пайплайн: **~1.9 мс GPU/кадр** при источнике 3456×2234. Память процесса — сотни МБ, не гигабайты.

### Бандл-модели (`Contents/Resources/Models`)

| Модель | Вход | GPU | ANE | Комментарий |
|---|---|---|---|---|
| DAv2-Small-Fast-518x294 | 518×294 | 9.8 мс (102 fps) | 10.8 мс | самая быстрая |
| **DAv2-Small-Balanced-644x364** | 644×364 | 14.7 мс (68 fps) | 29 мс | по умолчанию |
| DAv2-Small-Quality-812x462 | 812×462 | 26.6 мс (38 fps) | 62 мс | максимум деталей |
| DA3-Small-644x364 | 644×364 | 11.6 мс (86 fps) | 39 мс | Depth Anything 3 Small (конвертация в `convert/`), выдаёт depth — инверсия включается автоматически |

16:9-сборки DAv2 сконвертированы из `depth-anything/Depth-Anything-V2-Small-hf` (скрипты в `convert/`),
численно совпадают с PyTorch (corr = 1.000). Depth Pro не включён — ~секунды на кадр, не real-time.

Свою модель можно добавить кнопкой «+» (любой Core ML с image-входом и depth-выходом; кладётся в
`~/Library/Application Support/Real3D/Models`).

## Сборка (только Command Line Tools, Xcode не нужен)

```bash
./Scripts/build_app.sh          # → dist/Real3D.app + dist/Real3D.dmg
NO_DMG=1 ./Scripts/build_app.sh # только .app
swift build -c release --product real3d-bench   # CLI-бенчмарк
```

Шейдеры компилируются в рантайме (нет `metal`-тулчейна в CLT), Core ML-пакеты компилируются
в `.mlmodelc` скриптом при сборке. Подпись — Apple Development-сертификатом, если он есть в Keychain
(так разрешение Screen Recording не сбрасывается между сборками).

### CLI

```bash
.build/release/real3d-bench <model> --iters 60 --units all,ane,gpu --image photo.png --dump depth.png
.build/release/real3d-bench --stereo photo.png <model> out_prefix [strength] [convergence]   # офлайн-прогон всего пайплайна
.build/release/real3d-bench --shaders   # проверка компиляции шейдеров
```

## Использование

1. Запустить `Real3D.app`, разрешить Screen & System Audio Recording (System Settings → Privacy & Security).
2. Выбрать источник: дисплей или окно (полноэкранные приложения из других Spaces помечены «full screen»).
3. **Start** (⌘R). В превью появится рамка 16:9: тащить — перемещение, углы — размер. Пресеты
   ширины `1920/2560/3840` px и поля X/Y/W. Кроп запоминается для каждого источника.
4. Подключить XREAL, включить в очках 3D-режим (macOS увидит дисплей 3840×1080). В блоке **Output**
   выбрать этот дисплей, формат «3D Full SBS», нажать **⌘F** — окно вывода займёт дисплей очков.
   Esc — выход из полноэкранного режима. Если очки дают 1920×1080 — формат «Half SBS».
5. Настройки 3D: Depth strength (макс. диспаритет в px), Convergence (плоскость экрана),
   Depth curve, Temporal smoothing, Edge-aware upsampling, Hole-fill threshold. Без очков
   результат виден в нижнем превью (Anaglyph / Left / Right / Disparity map).

Лог: `~/Library/Logs/Real3D.log`.

### Замечания

- Приложение в **невидимом** Space (например, Chrome на весь экран, когда вы переключились на другой рабочий стол
  того же дисплея) перестаёт перерисовываться — macOS считает его скрытым, и кадры не приходят. Держите источник
  на одном дисплее, а Real3D и окно вывода — на другом (очки), тогда оба Space видимы.
- Подпись Apple Development, без нотаризации: на этом Mac запускается, на чужом Gatekeeper потребует
  «Open Anyway» (или Developer ID + notarize).

## Структура

```
Sources/Real3DCore/   DepthEngine (Core ML, zero-copy I/O), ScreenCapture (SCK), StereoRenderer (Metal),
                      Pipeline (display link + inference queue), Shaders.metal (→ ShaderSource.swift)
Sources/Real3D/       SwiftUI: AppModel, MainView, CropEditor, OutputWindow, MetalView
Sources/real3d-bench/ CLI: инспекция/бенчмарк моделей, офлайн-прогон стерео
Scripts/              build_app.sh, compile_models.swift, gen_shader_source.sh, make_icon.py
models/bundled/       .mlpackage, попадающие в бандл
convert/              скрипты конвертации 16:9-моделей (Python, отдельный conda env real3d-convert)
```
