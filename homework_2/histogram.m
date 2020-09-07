clear all
# https://octave.org/doc/v4.2.0/Multiple-Plots-on-One-Page.html

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
image = imread('lena.jpg');
# image = mat2gray(image);
raw_hist = imhist(image);
# figure(1)
# imshow(image)

eq_image = histeq(image);
eq_hist = imhist(eq_image);

# figure(2)
# imshow(eq_image)

# EXERCICIO 2
function my_image = MyHistoEq(img)
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
endfunction

my_eq_image = MyHistoEq(image)
# figure(3)
# imshow(eq_image)