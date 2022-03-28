1. Выбранный фильтр

Автоконтраст
Необходимо:
- Рассчитать яркость пикселей Y =  0.2125 * R + 0.7154 * G + 0.0721 * B.
- Построить гистограмму яркости на изображении.
- На основе гистограммы яркостей вычислить параметры растяжения яркости всего изображения.
- Применить полученные параметры растяжения яркости к исходному изображению.

2. Детали сборки проекта.

Используемое API - CUDA (собирал с CUDA версии 11.6).

Набор библиотек:
- stb для чтения/записи изображений (уже лежит в директории external)
- cmake

Порядок сборки:
mkdir build && cd build && cmake .. && make

3. Запуск приложения

./main input_image_path n_runs

- input_image_path - путь до изображения
- n_runs - число запусков фильтра для рассчета среднего времени выполнения

4. Спецификация ПК

Memory: 16 Gb
Processor: Intel® Core™ i7-9750H CPU @ 2.60GHz × 12
Graphics: NVIDIA Corporation TU116M [GeForce GTX 1660 Ti Mobile]

5. Результаты замеров времени на данном ПК (в миллисекундах, среднее по 20 запускам)

Замеры для CornellBox/Animation01_LDR_0001.png:
CPU time (1 thread): 10.9489
CPU time (8 threads): 2.78242
GPU time without copy: 0.210253
GPU time with copy: 1.00856
GPU copy time: 0.798304

Замеры для Bathroom01/Bathroom_LDR_0001.png:
CPU time (1 thread): 192.662
CPU time (8 threads): 45.5023
GPU time without copy: 2.75448
GPU time with copy: 13.7854
GPU copy time: 11.0309

Замеры для WasteWhite/WasteWhite_LDR_0001.png:
CPU time (1 thread): 30.8483
CPU time (8 threads): 7.80249
GPU time without copy: 0.592147
GPU time with copy: 2.71484
GPU copy time: 2.12269

6. Отчет о выполненных оптимизациях, полученном ускорении

6.1. Оптимизация вычисления гистограммы.

Простейшее ядро:

__global__ void histogram(uchar *y_img, uint *hist, int size)
{ 
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    atomicAdd(hist + y_img[i], 1);
}

Оптимизация:

Будем вычислять гитрограмму иерархически в 2-х ядрах.

1 ядро (block_histograms)

В shared памяти храним гистрограмму для каждой нити:
(длина гистограммы) * (число нитей) = 256 * 192 байт = 48 Кб
При этом тип гитрограммы unsigned char, поэтому каждая нить обрабатыеат не более 255 пикселей.

Затем суммируем все гитограммы в пределах блока.

2 ядро (merge_block_histograms)

Суммируем все гитрограммы блоков, полученные на предыдущем шаге.

Полученное ускорение (без учета копирования данных):

CornellBox/Animation01_LDR_0001.png: 2.22 раза
Bathroom01/Bathroom_LDR_0001.png: 1.85 раз
WasteWhite/WasteWhite_LDR_0001.png: 1.87 раз
