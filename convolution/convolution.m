clear all
close all
raw_img = imread('pirate.png');
raw_img = im2double(raw_img);

[h w] = size(raw_img);
kernel = [
-1 -1 -1;
-1 8 -1;
-1 -1 -1
];
modified_img = raw_img;

for i = 2 : h-1
  for j = 2 : w-1
    aux = raw_img(i-1:i+1, j-1:j+1) ;
    # at the folowing operation we use dot "." to multiply element-by-element.
    aux = abs(sum((aux .* kernel)(:))/size(kernel)(1)^2);
    # we can make a threshold to set aux with an condition.
    modified_img(i,j) = aux;
  endfor
endfor

figure(1)
subplot(1,2,1)
imshow(raw_img)
axis('image')
title('Raw image')

subplot(1,2,2)
imshow(modified_img)
title('Modified image')