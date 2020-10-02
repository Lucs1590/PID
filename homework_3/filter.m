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
