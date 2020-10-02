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


function near = getNear(list_near, k)
  i = 1;
  near = [];
  while (size(near)(2) < k)
    idx = nthargout (2, @min, abs (list_near-25)(:));
    [r, c] = ind2sub (size (list_near), idx);
    near(i) = list_near(r,c);
    list_near(r,c) = list_near(r,c) + 256;
    i++;
  endwhile
endfunction;

k_img = raw_img;
k = 9;
[h w] = size(raw_img);

for i = 3 : h-2
  for j = 3 : w-2
    aux = raw_img(i-2:i+2, j-2:j+2) ;
    aux = getNear(aux, k);
    k_img(i,j) =  sum(sum(aux))/k;
  endfor
endfor

figure('Name', 'Media da vizinhança k=9')
imshow(k_img)

k_img = raw_img;
k = 15;

for i = 3 : h-2
  for j = 3 : w-2
    aux = raw_img(i-2:i+2, j-2:j+2) ;
    aux = getNear(aux, k);
    k_img(i,j) =  sum(sum(aux))/k ;
  endfor
endfor

figure('Name', 'Media da vizinhança k=15')
imshow(k_img)

k_img = raw_img;
k = 20;

for i = 3 : h-2
  for j = 3 : w-2
    aux = raw_img(i-2:i+2, j-2:j+2) ;
    aux = getNear(aux, k);
    k_img(i,j) =  sum(sum(aux))/k ;
  endfor
endfor

figure('Name', 'Media da vizinhança k=20')
imshow(k_img)

# T_4
raw_img = imread('Figuras2/Moon.tif');
w4 = fspecial('laplacian',0);
w8 = [
1 1 1;
1 -8 1;
1 1 1
];
raw_img = im2double(raw_img);
img_lapacian = raw_img - imfilter(raw_img, w4, 'replicate');
img_kernel = raw_img - imfilter(raw_img, w8, 'replicate');
figure, imshow(raw_img)
figure, imshow(img_lapacian)
figure, imshow(img_kernel)

# E_10
img_6 = imread('Figuras2/ferramentas.bmp');
img_7 = imread('Figuras2/Lena_ruido.bmp');
img_8 = imread('Figuras2/Moon.tif');
img_9 = imread('Figuras2/teste.bmp');

sobel_v = fspecial('sobel');
sobel_h = fspecial('sobel')';
sobel_c = sqrt(sobel_v.^2 + sobel_h.^2)

img_result = img_6 - imfilter(img_6, sobel_v, 'replicate');
figure, imshow(img_result)
img_result = img_6 - imfilter(img_6, sobel_h, 'replicate');
figure, imshow(img_result)
img_result = img_6 - imfilter(img_6, sobel_c, 'replicate');
figure, imshow(img_result)

img_result = img_7 - imfilter(img_7, sobel_v, 'replicate');
figure, imshow(img_result)
img_result = img_7 - imfilter(img_7, sobel_h, 'replicate');
figure, imshow(img_result)
img_result = img_7 - imfilter(img_7, sobel_c, 'replicate');
figure, imshow(img_result)

img_result = img_8 - imfilter(img_8, sobel_v, 'replicate');
figure, imshow(img_result)
img_result = img_8 - imfilter(img_8, sobel_h, 'replicate');
figure, imshow(img_result)
img_result = img_8 - imfilter(img_8, sobel_c, 'replicate');
figure, imshow(img_result)

img_result = img_9 - imfilter(img_9, sobel_v, 'replicate');
figure, imshow(img_result)
img_result = img_9 - imfilter(img_9, sobel_h, 'replicate');
figure, imshow(img_result)
img_result = img_9 - imfilter(img_9, sobel_c, 'replicate');
figure, imshow(img_result)

prewitt_v = fspecial('prewitt');
prewitt_h = fspecial('prewitt')';
prewitt_c = sqrt(prewitt_v.^2 + prewitt_h.^2)

img_result = img_6 - imfilter(img_6, prewitt_v, 'replicate');
figure, imshow(img_result)
img_result = img_6 - imfilter(img_6, prewitt_h, 'replicate');
figure, imshow(img_result)
img_result = img_6 - imfilter(img_6, prewitt_c, 'replicate');
figure, imshow(img_result)

img_result = img_7 - imfilter(img_7, prewitt_v, 'replicate');
figure, imshow(img_result)
img_result = img_7 - imfilter(img_7, prewitt_h, 'replicate');
figure, imshow(img_result)
img_result = img_7 - imfilter(img_7, prewitt_c, 'replicate');
figure, imshow(img_result)

img_result = img_8 - imfilter(img_8, prewitt_v, 'replicate');
figure, imshow(img_result)
img_result = img_8 - imfilter(img_8, prewitt_h, 'replicate');
figure, imshow(img_result)
img_result = img_8 - imfilter(img_8, prewitt_c, 'replicate');
figure, imshow(img_result)

img_result = img_9 - imfilter(img_9, prewitt_v, 'replicate');
figure, imshow(img_result)
img_result = img_9 - imfilter(img_9, prewitt_h, 'replicate');
figure, imshow(img_result)
img_result = img_9 - imfilter(img_9, prewitt_c, 'replicate');
figure, imshow(img_result)

# T_5
k = imread('Figuras2/ferramentas.bmp');
f = im2double(k);
f1 = imnoise(f,'salt & pepper', 0.6);
f2 = imnoise(f,'salt & pepper', 0.6);
f3 = imnoise(f,'salt & pepper', 0.6);
f4 = imnoise(f,'salt & pepper', 0.6);
f5 = imnoise(f,'salt & pepper', 0.6);
f6 = imnoise(f,'salt & pepper', 0.6);
f7 = imnoise(f,'salt & pepper', 0.6);
f8 = imnoise(f,'salt & pepper', 0.6);
f9 = imnoise(f,'salt & pepper', 0.6);
f10 = imnoise(f,'salt & pepper', 0.6);

fm = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10)/10;

figure
subplot(2,3,1);imshow(f)
subplot(2,3,2);imshow(f1)
subplot(2,3,3);imshow(f2)
subplot(2,3,4);imshow(f3)
subplot(2,3,5);imshow(f4)
subplot(2,3,6);imshow(f5)

figure
subplot(2,3,1);imshow(f6)
subplot(2,3,2);imshow(f7)
subplot(2,3,3);imshow(f8)
subplot(2,3,4);imshow(f9)
subplot(2,3,5);imshow(f10)
subplot(2,3,6);imshow(fm)

# E_11
k = imread('Figuras2/ferramentas.bmp');
f = im2double(k);
f1 = imnoise(f,'salt & pepper', 0.9);
f2 = imnoise(f,'salt & pepper', 0.8);
f3 = imnoise(f,'salt & pepper', 0.7);
f4 = imnoise(f,'salt & pepper', 0.6);
f5 = imnoise(f,'salt & pepper', 0.5);
f6 = imnoise(f,'salt & pepper', 0.4);
f7 = imnoise(f,'salt & pepper', 0.3);
f8 = imnoise(f,'salt & pepper', 0.2);
f9 = imnoise(f,'salt & pepper', 0.1);
f10 = imnoise(f,'salt & pepper', 0.5);

fm = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10)/10;

figure
subplot(2,3,1);imshow(f)
subplot(2,3,2);imshow(f1)
subplot(2,3,3);imshow(f2)
subplot(2,3,4);imshow(f3)
subplot(2,3,5);imshow(f4)
subplot(2,3,6);imshow(f5)

figure
subplot(2,3,1);imshow(f6)
subplot(2,3,2);imshow(f7)
subplot(2,3,3);imshow(f8)
subplot(2,3,4);imshow(f9)
subplot(2,3,5);imshow(f10)
subplot(2,3,6);imshow(fm)

k = imread('Figuras2/ferramentas.bmp');
f = im2double(k);
f1 = imnoise(f,'gaussian', 0.9);
f2 = imnoise(f,'gaussian', 0.8);
f3 = imnoise(f,'gaussian', 0.7);
f4 = imnoise(f,'gaussian', 0.6);
f5 = imnoise(f,'gaussian', 0.5);
f6 = imnoise(f,'gaussian', 0.4);
f7 = imnoise(f,'gaussian', 0.3);
f8 = imnoise(f,'gaussian', 0.2);
f9 = imnoise(f,'gaussian', 0.1);
f10 = imnoise(f,'gaussian', 0.5);

fm = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10)/10;

figure
subplot(2,3,1);imshow(f)
subplot(2,3,2);imshow(f1)
subplot(2,3,3);imshow(f2)
subplot(2,3,4);imshow(f3)
subplot(2,3,5);imshow(f4)
subplot(2,3,6);imshow(f5)

figure
subplot(2,3,1);imshow(f6)
subplot(2,3,2);imshow(f7)
subplot(2,3,3);imshow(f8)
subplot(2,3,4);imshow(f9)
subplot(2,3,5);imshow(f10)
subplot(2,3,6);imshow(fm)

hist(fm)
fm = histeq(fm);
figure
imshow(fm)
