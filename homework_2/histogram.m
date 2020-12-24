clear all
close all

# 1) Faça um programa em Matlab, Octave ou Python que lê uma imagem, aplica a 
# transformação de equalização de histograma disponibilizado em bibliotecas de
# processamento de imagens da linguagem adotada e exibe:
# - imagem inicial;
# - imagem equalizada;
# - histograma da imagem original;
# - histograma de imagem equalizada;
# 2) Implemente uma função, MyHistoEq, que recebe uma imagem de entrada,
# calcula a equalização de histograma e retorna a imagem de entrada equalizada;
# 3) Repita 1), porém utilizando a função MyHistoEq
# 4) Compare os resultados

# EXERCICIO 1
image = imread('images/lena.jpg');
raw_hist = imhist(image);

figure(1)
subplot(2,2,1)
imshow(image)
axis('image')
title('Raw image')

subplot(2,2,3)
plot(raw_hist)
title('Raw histogram')

eq_image = histeq(image);
eq_hist = imhist(eq_image);

subplot(2,2,2)
imshow(eq_image)
title('Eq. image (Octave)')

subplot(2,2,4)
plot(eq_hist)
title('Eq. histogram (Octave)')

# EXERCICIO 2
function [my_image, my_histogram] = MyHistoEq(img)
  [h w] = size(img);
  flat_image = img(:);
  aux = zeros(256,1);
  
  # making a histogram
  for i = 1:256
    aux(i,1) = sum(flat_image == i) / size(flat_image)(1);
  endfor;

  # equalizing histogram
  bef = 0;
  for i = 1:256
    aux(i,1) = aux(i,1) + bef;
    bef = aux(i,1);
  endfor;
  
  for i = 1:256
    aux(i,1) = round(256 * aux(i,1));
  endfor;
  
  # equalizing image
  for j = 1:h
    for k = 1:w
      img(j,k) = aux(img(j,k),1);
    endfor;
  endfor;

  # return equalized image
  my_image = img;
  
  # making a new histogram
  flat_image = my_image(:);
  for i = 1:256
    aux(i,1) = sum(flat_image == i);
  endfor;
  my_histogram = aux;
endfunction

# EXERCICIO 3
figure(2)
subplot(2,2,1)
imshow(image)
axis('image')
title('Raw image')

subplot(2,2,3)
plot(raw_hist)
title('Raw histogram')

[my_eq_image, my_eq_histo] = MyHistoEq(image);

subplot(2,2,2)
imshow(my_eq_image)
title('Eq. image (Of authorship)')

subplot(2,2,4)
plot(my_eq_histo)
title('Eq. histogram (Of authorship)')
