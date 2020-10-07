close all
clear all
# E_1 e E_2
img = imread('Figuras2/mammogram.bmp');
function my_histogram = MyHisto(img)
  [h w] = size(img);
  flat_image = img(:);
  aux = zeros(256,1);
  
  for i = 1:256
    aux(i,1) = sum(flat_image == i) / size(flat_image)(1);
  endfor;
  my_histogram = aux;
endfunction;
# hist = MyHisto(img);
hist = imhist(img, 256);
figure()
# subplot(1,3,1);

bar(hist,0.1);
title("Bar histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 500 0 10000])


figure()
# subplot(1,3,2);
stem(hist, "linewidth", 0.1);
title("Stem histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 350 0 10500])

figure()
# subplot(1,3,3);
plot(hist)
title("Curve continues histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 300 0 10001])

pause

# E_3
figure()
g1 = imadjust(img, [0 1], [1 0]); # inversa
imshow(g1)

figure()
g2 = imadjust(img, [0.5 0.75], [0 1]); # do 0 ao 0.5, recebe zero, do 0.5 ao 0.75 está entre 0 e 1, e acima de 0.75 recebe 1
imshow(g2)

figure()
g3 = imadjust(img, [ ], [ ], 2); # If gamma is greater than 1, then imadjust weights the mapping toward lower (darker) output values.
imshow(g3)

pause

# T_2 e E_4
img2 = imread('Figuras2/polem.bmp');
figure()
imshow(img2)

figure()
imhist(img2)
ylim('auto')

img2_eq = histeq(img2,256);
figure()
imshow(img2_eq)

figure()
imhist(img2_eq)
ylim('auto')

pause

# E_5
figure()
imshow(img2)

figure()
imhist(img2)
ylim('auto')

hnorm = imhist(img2)./numel(img2);
cdf = cumsum(hnorm);
x = linspace(0, 1, 256);
figure()
plot(x,cdf)
axis([0 1 0 1])

set(gca, 'xtick', 0:.2:1)
set(gca, 'ytick', 0:.2:1)
xlabel('Valores de Intensidade de Entrada', 'fontsize', 9)
ylabel('Valores de Intensidade de Saída', 'fontsize', 9)
text(0.18, 0.5, 'Função de Transfomação', 'fontsize', 9)

pause

# E_6
img1 = imread('Figuras2/Image2.bmp');
img2 = imread('Figuras2/Image3.bmp');
img3 = imabsdiff(img2,img1);
img4 = imcomplement(img3);
img5 = img2 - img1;
figure('Name', 'Subtração B e A Comum'),imshow(img5)
img6 = img1 - img2;
figure('Name', 'Subtração A e B Comum'),imshow(img6)

pause

# E_7
img_polem = imread('Figuras2/polem.bmp');
figure('Name', 'Imagem Original')
imshow(img_polem)
img_polem_clara = img_polem + 128;
figure('Name', 'Imagem clareada')
imshow(img_polem_clara)

img_p_eq = histeq(img_polem,128);
img_pc_eq = histeq(img_polem_clara,128);
figure('Name', 'Histograma da Imagem Original')
imhist(img_p_eq)
figure('Name', 'Histograma da Imagem Clara')
imhist(img_pc_eq)

img_polem = imread('Figuras2/polem.bmp');
figure('Name', 'Imagem Original')
imshow(img_polem)
img_polem_clara = img_polem + 200;
figure('Name', 'Imagem clareada')
imshow(img_polem_clara)

img_p_eq = histeq(img_polem,128);
img_pc_eq = histeq(img_polem_clara,128);
figure('Name', 'Histograma da Imagem Original')
imhist(img_p_eq)
figure('Name', 'Histograma da Imagem Clara')
imhist(img_pc_eq)

figure('Name', 'Imagem Original Equalizada')
imshow(img_p_eq)
figure('Name', 'Imagem Clara Equalizada')
imshow(img_p_eq)

pause

# E_8
raw_img = imread('Figuras2/ferramentas.bmp');
img_bin_1 = im2bw(raw_img,0.2);
img_bin_2 = im2bw(raw_img,0.3);
img_bin_3 = im2bw(raw_img,0.4);
img_bin_4 = im2bw(raw_img,0.5);
img_bin_5 = im2bw(raw_img,0.6);
figure('Name', 'Imagem Bin 20%')
imshow(img_bin_1)
figure('Name', 'Imagem Bin 30%')
imshow(img_bin_2)
figure('Name', 'Imagem Bin 40%')
imshow(img_bin_3)
figure('Name', 'Imagem Bin 50%')
imshow(img_bin_4)
figure('Name', 'Imagem Bin 60%')
imshow(img_bin_5)

figure('Name', 'Histograma do Bin 20%')
imhist(img_bin_1)
figure('Name', 'Histograma Bin 30%')
imhist(img_bin_2)
figure('Name', 'Histograma Bin 40%')
imhist(img_bin_3)
figure('Name', 'Histograma Bin 50%')
imhist(img_bin_4)
figure('Name', 'Histograma Bin 60%')
imhist(img_bin_5)

raw_img = imread('Figuras2/polem.bmp');
img_bin_6 = graythresh(raw_img)
figure('Name', 'Imagem com Graythresh')
imshow(img_bin_6)
figure('Name', 'Histograma Graythresh')
imhist(img_bin_6)

pause

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

pause

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

pause

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
