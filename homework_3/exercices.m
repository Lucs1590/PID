clear all
close all

# T_1
f = imread('Figuras2/mammogram.bmp');
imfinfo Figuras2/mammogram.bmp;
figure(1)
subplot(1,2,1);
imshow(f);
subplot(1,2,2);
imhist(f);

# E_1
