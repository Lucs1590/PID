clear all
# https://octave.org/doc/v4.4.1/Basic-Statistical-Functions.html
# https://octave.org/doc/v4.0.1/Defining-Functions.html#Defining-Functions

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
image = mat2gray(image);
raw_hist = imhist(image);
# figure(1)
# imshow(image)

eq_image = histeq(image);
eq_hist = imhist(eq_image);
