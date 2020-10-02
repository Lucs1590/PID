clear all
close all
# E_9
raw_img = imread('Figuras2/Lena_ruido.bmp');
img_mean_5 = imfilter(raw_img, fspecial("average", 7));
img_mean_7 = imfilter(raw_img, fspecial("average", 7));
img_mean_9 = imfilter(raw_img, fspecial("average", 9));
img_mean_25 = imfilter(raw_img, fspecial("average", 25));
img_mean_31 = imfilter(raw_img, fspecial("average", 31));

figure('Name', 'Media 5x5')
imshow(img_mean_5)
figure('Name', 'Media 7x7')
imshow(img_mean_7)
figure('Name', 'Media 9x9')
imshow(img_mean_9)
figure('Name', 'Media 25x25')
imshow(img_mean_25)
figure('Name', 'Media 31x31')
imshow(img_mean_31)

img_med_5 = medfilt2(raw_img, [5 5]);
img_med_7 = medfilt2(raw_img, [7 7]);
figure('Name', 'Mediana 5x5')
imshow(img_med_5)
figure('Name', 'Mediana 7x7')
imshow(img_med_7)

