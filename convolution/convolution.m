clear all
close all
raw_img = imread('pirate.png');
raw_img = im2double(raw_img);

[h w] = size(raw_img);

dot_kernel = [
-1 -1 -1;
-1 8 -1;
-1 -1 -1
];

vertical_kernel = [
-1 2 -1;
-1 2 -1;
-1 2 -1
];

horizontal_kernel = [
-1 -1 -1;
2 2 2;
-1 -1 -1
];

diagonal_r_kernel = [
-1 -1 2;
-1 2 -1;
2 -1 -1
];

diagonal_l_kernel = [
2 -1 -1;
-1 2 -1;
-1 -1 2
];


dot_image = raw_img;
vertical_image =  raw_img;
horizontal_image  = raw_img;
diagonal_r_image = raw_img;
diagonal_l_image = raw_img;

for i = 2 : h-1
  for j = 2 : w-1
    # at the folowing operation we use dot "." to multiply element-by-element.
    # we can make a threshold to set aux with an condition.
    dot_image(i,j) = abs(sum((raw_img(i-1:i+1, j-1:j+1) .* dot_kernel)(:))/size(dot_kernel)(1)^2);
    vertical_image(i,j) = abs(sum((raw_img(i-1:i+1, j-1:j+1) .* vertical_kernel)(:))/size(vertical_kernel)(1)^2);
    horizontal_image(i,j) = abs(sum((raw_img(i-1:i+1, j-1:j+1) .* horizontal_kernel)(:))/size(horizontal_kernel)(1)^2);
    diagonal_r_image(i,j) = abs(sum((raw_img(i-1:i+1, j-1:j+1) .* diagonal_r_kernel)(:))/size(diagonal_r_kernel)(1)^2);
    diagonal_l_image(i,j) = abs(sum((raw_img(i-1:i+1, j-1:j+1) .* diagonal_l_kernel)(:))/size(diagonal_l_kernel)(1)^2);
  endfor
endfor

figure(1)
subplot(2,3,1)
imshow(raw_img)
axis('image')
title('Raw image')

subplot(2,3,2)
imshow(dot_image)
title('Central Dot Kernel')

subplot(2,3,3)
imshow(vertical_image)
title('Vertical Kernel')

subplot(2,3,4)
imshow(horizontal_image)
title('Horizontal Kernel')

subplot(2,3,5)
imshow(diagonal_r_image)
title('Diagonal Right Kernel')

subplot(2,3,6)
imshow(diagonal_l_image)
title('Diagonal Right Kernel')